#!/bin/bash
# Push current state to Hugging Face Spaces cleanly (no binary history)
set -e

cleanup() {
  git checkout main 2>/dev/null || true
  git branch -D hf-tmp 2>/dev/null || true
}

echo "→ Creating clean snapshot for HF..."
git checkout --orphan hf-tmp
git rm -rf --cached . > /dev/null 2>&1
git add .

BINARIES=$(git diff --cached --name-only | grep -E "\.(png|jpg|pt|pth)$" || true)
if [ -n "$BINARIES" ]; then
  echo "⚠ Warning: these binaries are still staged:"
  echo "$BINARIES"
  cleanup
  exit 1
fi

git commit -m "deploy: $(date '+%Y-%m-%d %H:%M')"
echo "→ Force-pushing to hf/main..."
git push hf hf-tmp:main --force
cleanup
echo "✓ Done! Space is updating at https://huggingface.co/spaces/DariusGiannoli/PerceptionBenchmark"
