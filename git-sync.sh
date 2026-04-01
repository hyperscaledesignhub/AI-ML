#!/bin/bash

# Git Auto-Sync Script
# This script handles common git sync issues automatically

set -e  # Exit on any error

echo "🔄 Starting Git Auto-Sync..."

# Navigate to the repository directory
REPO_DIR="/Users/vijayabhaskarv/IOT/github/AI-ML"
cd "$REPO_DIR"

echo "📍 Working in: $(pwd)"

# Check if there are any uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "💾 Found uncommitted changes. Committing them..."
    git add .
    git commit -m "Auto-commit: Save work in progress - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "✅ Changes committed"
else
    echo "✅ Working tree is clean"
fi

# Check if we're ahead, behind, or diverged
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "no-remote")
BASE=$(git merge-base @ @{u} 2>/dev/null || echo "no-base")

if [ "$REMOTE" = "no-remote" ]; then
    echo "❌ No remote tracking branch found"
    exit 1
fi

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "✅ Already up to date"
elif [ "$LOCAL" = "$BASE" ]; then
    echo "⬇️  Behind remote. Pulling changes..."
    git pull
    echo "✅ Successfully pulled changes"
elif [ "$REMOTE" = "$BASE" ]; then
    echo "⬆️  Ahead of remote. Pushing changes..."
    if ! git push 2>/dev/null; then
        echo "❌ Push failed. Remote has new changes. Fetching and merging..."
        git fetch
        git pull --no-rebase
        echo "✅ Successfully merged remote changes"
        echo "⬆️  Pushing merged changes..."
        git push
        echo "✅ Successfully pushed merged changes"
    else
        echo "✅ Successfully pushed changes"
    fi
else
    echo "🔀 Branches have diverged. Merging..."
    # Set pull strategy to merge (not rebase)
    git config pull.rebase false
    git pull --no-rebase
    echo "✅ Successfully merged remote changes"
    
    echo "⬆️  Pushing merged changes..."
    git push
    echo "✅ Successfully pushed merged changes"
fi

echo "🎉 Git sync completed successfully!"
echo "📊 Current status:"
git status --porcelain
