#!/usr/bin/env bash
#
# Local CI. Runs every quality gate the project defines and prints one verdict.
#
# Three things distinguish this from `make check`:
#
#   1. It keeps going after a failure, so one run reports every problem rather
#      than only the first.
#   2. A gate whose toolchain is absent is reported BLOCKED and fails the run.
#      Silence would let a missing Node install read as a green frontend, which
#      is exactly the "claimed a pass" failure AGENTS.md rules out.
#   3. It checks the lockfiles, which `make check` does not: a resolution that
#      only works because of what is already installed locally is a real defect
#      and the cheapest possible thing to catch.
#
# Usage:
#   scripts/ci-local.sh             every gate
#   scripts/ci-local.sh backend     backend gates only
#   scripts/ci-local.sh frontend    frontend gates only
#
# Exit status:
#   0  every selected gate passed
#   1  at least one gate failed
#   2  at least one gate could not run at all
#
# The complete CPU release image runs in pull-request CI. Local CI validates all
# Compose variants; the multi-gigabyte CUDA build remains a scheduled gate.

# No `set -e`: a failing gate must be recorded and the run continued.
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
cd -- "${repository_root}" || exit

selection="${1:-all}"
case "${selection}" in
  all | backend | frontend) ;;
  *)
    echo "usage: $(basename "$0") [all|backend|frontend]" >&2
    exit 64
    ;;
esac

if [[ -t 1 ]]; then
  bold=$'\033[1m'; dim=$'\033[2m'; red=$'\033[31m'
  green=$'\033[32m'; yellow=$'\033[33m'; reset=$'\033[0m'
else
  bold=''; dim=''; red=''; green=''; yellow=''; reset=''
fi

log_dir="$(mktemp -d -t upscaler-ci-local.XXXXXX)"
trap 'rm -rf -- "${log_dir}"' EXIT

names=()
states=()
notes=()
failed=0
blocked=0

record() {
  names+=("$1")
  states+=("$2")
  notes+=("$3")
}

# A gate that cannot run is never silently dropped: it is recorded, it prints
# what would fix it, and it fails the run. The short note keeps the summary
# scannable; the full reason is printed once, here.
skip_gate() {
  local name="$1" note="$2" reason="$3"
  printf '%s %s\n' "${yellow}BLOCKED${reset}" "${bold}${name}${reset}"
  printf '        %s\n' "${reason}"
  record "${name}" BLOCKED "${note}"
  blocked=1
}

run_gate() {
  local name="$1"
  shift
  local log="${log_dir}/${name//[^A-Za-z0-9]/-}.log"
  local start=${SECONDS}

  printf '%s %s %s\n' "${dim}····${reset}" "${bold}${name}${reset}" "${dim}$*${reset}"
  "$@" >"${log}" 2>&1
  local status=$?
  local elapsed=$(( SECONDS - start ))

  if (( status == 0 )); then
    printf '%s %s %s\n' "${green}PASS${reset}   " "${bold}${name}${reset}" "${dim}(${elapsed}s)${reset}"
    record "${name}" PASS "${elapsed}s"
    return 0
  fi

  printf '%s %s %s\n' "${red}FAIL${reset}   " "${bold}${name}${reset}" \
    "${dim}(${elapsed}s, exit ${status})${reset}"
  # The tail is the part that says what broke; the full log is kept below.
  sed 's/^/        /' "${log}" | tail -n 40
  record "${name}" FAIL "exit ${status}"
  failed=1
  return 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

audit_backend() {
  # Audit every optional engine as well as the lightweight environment we
  # install for tests. GPU dependencies can be inspected from the lockfile
  # without downloading their multi-gigabyte wheels.
  uv export --locked --all-extras --no-emit-project \
    | uv run pip-audit -r /dev/stdin --require-hashes --disable-pip \
      --progress-spinner off --cache-dir "${log_dir}/pip-audit-cache"
}

install_frontend_browser() {
  if [[ -n "${CI:-}" ]]; then
    pnpm --dir frontend exec playwright install --with-deps chromium
  else
    pnpm --dir frontend exec playwright install chromium
  fi
}

# nvm appends its init to the end of ~/.bashrc, below the guard that makes that
# file return early for non-interactive shells. A hook, a cron entry, or an
# agent shell therefore sees no Node even on a machine where it is installed,
# and would be told the frontend gates are BLOCKED when they are merely hidden.
# Loading it here costs nothing when Node is already on PATH.
load_nvm_if_needed() {
  local nvm_sh="${NVM_DIR:-${HOME}/.nvm}/nvm.sh"
  if ! have node && [[ -s "${nvm_sh}" ]]; then
    # shellcheck source=/dev/null
    . "${nvm_sh}" >/dev/null 2>&1 || true
  fi
}

printf '%s\n\n' "${bold}Local CI — ${selection} gates${reset}"

if [[ "${selection}" == all || "${selection}" == backend ]]; then
  printf '%s\n' "${bold}Backend${reset}"
  if ! have uv; then
    for gate in "backend deps" "backend lockfile" "backend audit" "backend lint" \
      "backend format" "backend types" "backend shell" "backend tests" "backend package"; do
      skip_gate "${gate}" "uv missing" \
        "Requires uv. Install from https://docs.astral.sh/uv/ and re-run."
    done
  else
    # Sync first: every later backend gate runs inside this environment, so a
    # broken sync would otherwise surface as four confusing failures.
    if run_gate "backend deps" uv sync --extra dev --locked; then
      run_gate "backend lockfile" uv lock --check
      run_gate "backend audit" audit_backend
      run_gate "backend lint" uv run ruff check backend scripts
      run_gate "backend format" uv run ruff format --check backend scripts
      run_gate "backend types" uv run mypy backend/upscaler
      run_gate "backend shell" uv run shellcheck scripts/*.sh
      run_gate "backend tests" uv run pytest --cov=upscaler --cov-branch --cov-report=term-missing
      run_gate "backend package" uv run python scripts/check-package.py
    else
      for gate in "backend lockfile" "backend audit" "backend lint" "backend format" \
        "backend types" "backend shell" "backend tests" "backend package"; do
        skip_gate "${gate}" "prerequisite failed" "Needs 'backend deps', which did not complete."
      done
    fi
  fi
  printf '\n'
fi

if [[ "${selection}" == all || "${selection}" == frontend ]]; then
  printf '%s\n' "${bold}Frontend${reset}"
  frontend_gates=("frontend deps" "frontend audit" "frontend lint" "frontend format" \
    "frontend types" "frontend tests" "frontend build" "frontend browser" "frontend e2e")
  load_nvm_if_needed
  missing=()
  if ! have node; then
    missing+=("Node.js 22.13 or newer")
  elif ! node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 22 || (major === 22 && minor >= 13) ? 0 : 1)'; then
    missing+=("Node.js 22.13 or newer (found $(node --version))")
  fi
  if ! have pnpm; then
    missing+=("pnpm 10.34.5 (https://pnpm.io/)")
  elif [[ "$(pnpm --version)" != "10.34.5" ]]; then
    missing+=("pnpm 10.34.5 (found $(pnpm --version))")
  fi

  if (( ${#missing[@]} )); then
    joined="${missing[0]}"
    for tool in "${missing[@]:1}"; do joined="${joined} and ${tool}"; done
    reason="Requires ${joined}."
    for gate in "${frontend_gates[@]}"; do
      skip_gate "${gate}" "toolchain missing" "${reason} Install it and re-run."
    done
  elif run_gate "frontend deps" pnpm --dir frontend install --frozen-lockfile; then
    run_gate "frontend audit" pnpm --dir frontend audit --audit-level high
    run_gate "frontend lint" pnpm --dir frontend lint
    run_gate "frontend format" pnpm --dir frontend format:check
    run_gate "frontend types" pnpm --dir frontend check
    run_gate "frontend tests" pnpm --dir frontend test:coverage
    run_gate "frontend build" pnpm --dir frontend build
    if run_gate "frontend browser" install_frontend_browser; then
      run_gate "frontend e2e" pnpm --dir frontend test:e2e
    else
      skip_gate "frontend e2e" "prerequisite failed" \
        "Needs 'frontend browser', which did not complete."
    fi
  else
    for gate in "frontend audit" "frontend lint" "frontend format" "frontend types" \
      "frontend tests" "frontend build" "frontend browser" "frontend e2e"; do
      skip_gate "${gate}" "prerequisite failed" "Needs 'frontend deps', which did not complete."
    done
  fi
  printf '\n'
fi

if [[ "${selection}" == all ]]; then
  printf '%s\n' "${bold}Container${reset}"
  if have docker && docker compose version >/dev/null 2>&1; then
    run_gate "compose config" make compose-config
  else
    skip_gate "compose config" "Docker missing" \
      "Requires Docker with the Compose plugin. Install it and re-run."
  fi
  printf '\n'
fi

printf '%s\n' "${bold}Summary${reset}"
for index in "${!names[@]}"; do
  case "${states[index]}" in
    PASS) marker="${green}PASS${reset}   " ;;
    FAIL) marker="${red}FAIL${reset}   " ;;
    *) marker="${yellow}BLOCKED${reset}" ;;
  esac
  printf '  %s %-18s %s\n' "${marker}" "${names[index]}" "${dim}${notes[index]}${reset}"
done
printf '\n'

if (( failed )); then
  printf '%s\n' "${red}${bold}FAILED${reset} — at least one gate did not pass."
  exit 1
fi
if (( blocked )); then
  # Not a pass. The unrun gates are unknown, not green, and saying otherwise
  # here would defeat the point of running this at all.
  printf '%s\n' "${yellow}${bold}INCOMPLETE${reset} — every gate that ran passed, but some could not run."
  printf '%s\n' "Install the tooling above, or run a subset deliberately with 'make ci-local GATES=backend'."
  exit 2
fi
printf '%s\n' "${green}${bold}PASSED${reset} — every quality gate is green."
