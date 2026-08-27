#!/usr/bin/env bash
# Smoke-test a built release image through its public HTTP interface.

set -euo pipefail

image="${1:-local-image-upscaler:cpu}"
container="upscaler-release-smoke-${RANDOM}"
temporary="$(mktemp -d -t upscaler-container-smoke.XXXXXX)"

cleanup() {
  docker rm --force --volumes "${container}" >/dev/null 2>&1 || true
  rm -rf -- "${temporary}"
}
trap cleanup EXIT

configured_user="$(docker image inspect --format '{{.Config.User}}' "${image}")"
if [[ "${configured_user}" != "10001:10001" ]]; then
  echo "container smoke: image user is '${configured_user}', expected 10001:10001" >&2
  exit 1
fi

docker run --rm --entrypoint sh "${image}" -c '
  test "$(id -u)" = 10001
  test -r /usr/share/licenses/local-image-upscaler/LICENSE
  test -r /usr/share/licenses/local-image-upscaler/NOTICE
  ! python3 -c "import torch" >/dev/null 2>&1
'

docker run --detach --name "${container}" \
  --mount type=volume,destination=/weights \
  --mount type=volume,destination=/work \
  --publish 127.0.0.1::8000 \
  "${image}" >/dev/null

port="$(docker inspect --format '{{(index (index .NetworkSettings.Ports "8000/tcp") 0).HostPort}}' "${container}")"
base_url="http://127.0.0.1:${port}"

healthy=0
for _attempt in $(seq 1 30); do
  if curl --fail --silent "${base_url}/api/v1/health" >"${temporary}/health.json"; then
    healthy=1
    break
  fi
  sleep 2
done
if (( ! healthy )); then
  docker logs "${container}" >&2
  exit 1
fi

curl --fail --silent "${base_url}/" | grep --quiet '<div id="root"></div>'

base64 --decode >"${temporary}/source.png" <<'PNG'
iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=
PNG

curl --fail --silent \
  --form "file=@${temporary}/source.png;type=image/png" \
  --form processing_mode=sharpen_only \
  --form sharpen=25 \
  "${base_url}/api/v1/jobs" >"${temporary}/created.json"
job_id="$(python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])' <"${temporary}/created.json")"

completed=0
for _attempt in $(seq 1 30); do
  curl --fail --silent "${base_url}/api/v1/jobs/${job_id}" >"${temporary}/job.json"
  state="$(python3 -c 'import json, sys; print(json.load(sys.stdin)["state"])' <"${temporary}/job.json")"
  case "${state}" in
    completed)
      completed=1
      break
      ;;
    failed | cancelled)
      echo "container smoke: job entered terminal state ${state}" >&2
      cat "${temporary}/job.json" >&2
      exit 1
      ;;
  esac
  sleep 1
done
if (( ! completed )); then
  echo "container smoke: job did not complete" >&2
  cat "${temporary}/job.json" >&2
  exit 1
fi

curl --fail --silent "${base_url}/api/v1/jobs/${job_id}/result" >"${temporary}/result.png"
python3 -c 'import pathlib, sys; assert pathlib.Path(sys.argv[1]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")' \
  "${temporary}/result.png"
curl --fail --silent --request DELETE "${base_url}/api/v1/jobs/${job_id}" >/dev/null

echo "container smoke passed: ${image}"
