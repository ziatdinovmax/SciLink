#! /bin/bash

# Fetch all branches from upstream
git fetch upstream

# Update main branch
git checkout main
git merge upstream/main
git push origin main

# Get a list of your local branches
branches=$(git branch | grep -v "main" | tr -d "* ")

# Update each branch
for branch in $branches; do
    echo "Updating branch $branch..."
    git checkout $branch
    git merge main  # or git rebase main
    git push origin $branch
done
