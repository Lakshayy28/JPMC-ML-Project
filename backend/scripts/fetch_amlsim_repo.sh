#!/usr/bin/env sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/data/external/AMLSim"
REPO_URL="https://github.com/IBM/AMLSim.git"

mkdir -p "$ROOT_DIR/data/external"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "AMLSim already present at $TARGET_DIR"
  exit 0
fi

git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
echo "Cloned AMLSim into $TARGET_DIR"