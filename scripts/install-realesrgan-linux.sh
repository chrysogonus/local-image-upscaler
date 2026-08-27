#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
install_dir="${UPSCALER_REALESRGAN_DIR:-${repository_root}/.upscaler/realesrgan-ncnn-vulkan}"
manifest_path="${repository_root}/models/manifest.json"
runtime_id="realesrgan-ncnn-vulkan-linux-x64"

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) ;;
  *)
    echo "This installer currently supports Linux x86_64 only." >&2
    exit 2
    ;;
esac

# The URL and digest come from models/manifest.json rather than being repeated
# here, so the manifest stays the single checksum authority: a pin bumped there
# cannot silently disagree with what this installer actually fetches. An entry
# with a null sha256 is refused, matching install-weights.py.
#
# Command substitution, not a process substitution: `set -e` aborts on a failed
# extraction here, whereas `read < <(...)` would swallow the error and leave the
# URL empty.
manifest_fields="$(
  python3 - "${manifest_path}" "${runtime_id}" <<'PYTHON'
import json
import sys

manifest_path, runtime_id = sys.argv[1], sys.argv[2]
try:
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
except OSError as error:
    sys.exit(f"cannot read {manifest_path}: {error.strerror}")
except json.JSONDecodeError as error:
    sys.exit(f"{manifest_path} is not valid JSON: {error}")

for entry in manifest.get("runtimes", []):
    if entry.get("id") == runtime_id:
        break
else:
    sys.exit(f"{runtime_id} is not declared in models/manifest.json")

url, digest = entry.get("archive_url"), entry.get("sha256")
if not url or not url.startswith("https://"):
    sys.exit(f"{runtime_id}: archive_url must be an https:// URL")
if not digest:
    sys.exit(
        f"{runtime_id}: no checksum is pinned in models/manifest.json.\n"
        "Verify the upstream release and record its sha256 before installing."
    )
print(url, digest)
PYTHON
)"
read -r archive_url expected_sha256 <<<"${manifest_fields}"

archive_path="$(mktemp --suffix=.zip)"
cleanup() {
  rm -f -- "${archive_path}"
}
trap cleanup EXIT

echo "Downloading the official Real-ESRGAN NCNN/Vulkan release..."
curl --fail --location --proto '=https' --tlsv1.2 "${archive_url}" --output "${archive_path}"
echo "${expected_sha256}  ${archive_path}" | sha256sum --check --status

mkdir -p -- "${install_dir}"
unzip -q -o "${archive_path}" \
  'models/*' \
  'realesrgan-ncnn-vulkan' \
  'README_ubuntu.md' \
  -d "${install_dir}"
chmod 0755 -- "${install_dir}/realesrgan-ncnn-vulkan"

echo "Installed and checksum-verified at ${install_dir}"
echo "Restart the local backend; it will detect this repository-local runtime automatically."
