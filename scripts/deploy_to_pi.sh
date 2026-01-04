#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/deployment/"
PI_HOST="${PI_HOST:-pi@10.0.0.51}"
PI_DEPLOY_DIR="${PI_DEPLOY_DIR:-/home/pi/harold}"

rsync -av --delete \
  --exclude "logs/" \
  --exclude "sessions/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "${SRC_DIR}" "${PI_HOST}:${PI_DEPLOY_DIR}/"

if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  COMMIT=$(git -C "${ROOT_DIR}" rev-parse --short HEAD)
  DIRTY_COUNT=$(git -C "${ROOT_DIR}" status --porcelain | wc -l | tr -d ' ')
  TS=$(date -Is)
  ssh "${PI_HOST}" "cat > ${PI_DEPLOY_DIR}/DEPLOYED_COMMIT.txt <<'EOF'
commit: ${COMMIT}
timestamp: ${TS}
worktree_dirty: ${DIRTY_COUNT}
EOF"
fi

printf "Deployed %s to %s:%s\n" "${SRC_DIR}" "${PI_HOST}" "${PI_DEPLOY_DIR}"
