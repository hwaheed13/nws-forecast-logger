#!/bin/bash
set -e

# --- CONFIG ---
REPO_DIR="/Users/hyderwaheed/nws-forecast-logger"
LOG_FILE="/Users/hyderwaheed/nws-forecast-logger/git_sync.log"
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
# ---------------

cd "$REPO_DIR"

# Make sure identity is set (safe if already set)
git config user.name "Hyder Waheed" >/dev/null
git config user.email "hwaheed13@gmail.com" >/dev/null

# 1) Pull latest from GitHub (rebase to avoid messy merges)
git pull --rebase origin main || true

# 2) Stage & commit only if there are changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  git add -A
  git commit -m "Daily local autosync [skip ci]" || true
  git push origin main || true
else
  echo "$(date) — Nothing to sync" >> "$LOG_FILE"
fi

