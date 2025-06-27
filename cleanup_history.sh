#!/bin/bash
# Script to clean sensitive data from git history

echo "⚠️  WARNING: This will rewrite git history!"
echo "Make sure you have a backup of your repository"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Remove sensitive files from all commits
git filter-branch --force --index-filter \
  'git rm -rf --cached --ignore-unmatch raw_data/*.csv data/processed/* output/* notebooks/*.csv eda/*.csv' \
  --prune-empty --tag-name-filter cat -- --all

# Clean up refs
git for-each-ref --format="%(refname)" refs/original/ | xargs -n 1 git update-ref -d

# Garbage collect
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "✅ History cleaned. To push changes:"
echo "git push origin --force --all"
echo "git push origin --force --tags"