#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STAMP="$(date -u +%F)"
OUT_PATH="${OUT_PATH:-$REPO_ROOT/reports/docker_base_images_${STAMP}.tsv}"

docker_cmd() {
  if [ "$(id -u)" -ne 0 ]; then
    sudo docker "$@"
  else
    command docker "$@"
  fi
}

mkdir -p "$(dirname "$OUT_PATH")"

declare -A arg_defaults=()
declare -A img_to_dockerfiles=()
declare -A seen_img=()
images=()

resolve_from_token() {
  local token="$1"
  # Replace ${VAR} or $VAR with ARG defaults when possible (simple, non-eval parsing).
  if [[ "$token" == '${'*'}' ]]; then
    local var="${token:2:${#token}-3}"
    if [ -n "${arg_defaults[$var]:-}" ]; then
      echo "${arg_defaults[$var]}"
      return
    fi
  fi
  if [[ "$token" == '$'* && "$token" != '${'* ]]; then
    local var="${token:1}"
    if [ -n "${arg_defaults[$var]:-}" ]; then
      echo "${arg_defaults[$var]}"
      return
    fi
  fi
  echo "$token"
}

while IFS= read -r dockerfile; do
  # Reset ARG defaults per Dockerfile (ARG scoping in Dockerfiles matters, but our usage is simple).
  arg_defaults=()

  while IFS= read -r line; do
    # Strip comments.
    line="${line%%#*}"
    line="$(echo "$line" | sed -E 's/[[:space:]]+$//')"
    [ -z "$line" ] && continue

    if [[ "$line" =~ ^ARG[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)=(.+)$ ]]; then
      key="${BASH_REMATCH[1]}"
      val="${BASH_REMATCH[2]}"
      val="$(echo "$val" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
      arg_defaults["$key"]="$val"
      continue
    fi

    if [[ "$line" =~ ^FROM[[:space:]]+([^[:space:]]+) ]]; then
      raw_token="${BASH_REMATCH[1]}"
      img="$(resolve_from_token "$raw_token")"
      if [ -z "${seen_img[$img]:-}" ]; then
        seen_img["$img"]=1
        images+=("$img")
      fi
      if [ -z "${img_to_dockerfiles[$img]:-}" ]; then
        img_to_dockerfiles["$img"]="$dockerfile"
      else
        img_to_dockerfiles["$img"]+=",${dockerfile}"
      fi
    fi
  done <"$dockerfile"
done < <(find "$REPO_ROOT" -maxdepth 4 -name 'Dockerfile*' -print | sort)

{
  echo -e "captured_at_utc\\timage\\tos\\tarch\\timage_id\\trepo_digests\\tcreated\\tsize_bytes\\tdockerfiles"
  for img in "${images[@]}"; do
    os="n/a"
    arch="n/a"
    image_id="missing"
    repo_digests="[]"
    created="n/a"
    size_bytes="n/a"

    if docker_cmd image inspect "$img" >/dev/null 2>&1; then
      os="$(docker_cmd image inspect "$img" --format '{{.Os}}' 2>/dev/null || echo n/a)"
      arch="$(docker_cmd image inspect "$img" --format '{{.Architecture}}' 2>/dev/null || echo n/a)"
      image_id="$(docker_cmd image inspect "$img" --format '{{.Id}}' 2>/dev/null || echo n/a)"
      repo_digests="$(docker_cmd image inspect "$img" --format '{{json .RepoDigests}}' 2>/dev/null || echo '[]')"
      created="$(docker_cmd image inspect "$img" --format '{{.Created}}' 2>/dev/null || echo n/a)"
      size_bytes="$(docker_cmd image inspect "$img" --format '{{.Size}}' 2>/dev/null || echo n/a)"
    else
      # Best-effort: pull small base images so we can record a resolved digest.
      case "$img" in
        rocm/*|vllm/*)
          ;;
        *)
          docker_cmd pull "$img" >/dev/null 2>&1 || true
          if docker_cmd image inspect "$img" >/dev/null 2>&1; then
            os="$(docker_cmd image inspect "$img" --format '{{.Os}}' 2>/dev/null || echo n/a)"
            arch="$(docker_cmd image inspect "$img" --format '{{.Architecture}}' 2>/dev/null || echo n/a)"
            image_id="$(docker_cmd image inspect "$img" --format '{{.Id}}' 2>/dev/null || echo n/a)"
            repo_digests="$(docker_cmd image inspect "$img" --format '{{json .RepoDigests}}' 2>/dev/null || echo '[]')"
            created="$(docker_cmd image inspect "$img" --format '{{.Created}}' 2>/dev/null || echo n/a)"
            size_bytes="$(docker_cmd image inspect "$img" --format '{{.Size}}' 2>/dev/null || echo n/a)"
          fi
          ;;
      esac
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$img" "$os" "$arch" "$image_id" "$repo_digests" "$created" "$size_bytes" \
      "${img_to_dockerfiles[$img]:-}" \
      | sed -E 's#/home/kj/strix-halo-inference#\\$REPO_ROOT#g'
  done
} >"$OUT_PATH"

echo "$OUT_PATH"
