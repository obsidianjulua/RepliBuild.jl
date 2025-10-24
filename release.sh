#!/bin/bash
# Release script for RepliBuild.jl v1.1.0

set -e  # Exit on error

echo "🚀 RepliBuild.jl v1.1.0 Release Script"
echo "======================================"
echo ""

# Check we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "⚠️  Warning: You're on branch '$BRANCH', not 'main'"
    echo "Switch to main? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        git checkout main
    else
        echo "Aborting release"
        exit 1
    fi
fi

# Delete old v1.2 tag if exists
echo "🧹 Cleaning up old tags..."
git tag -d v1.2 2>/dev/null && echo "   Deleted local v1.2 tag" || echo "   No v1.2 tag to delete"

# Show status
echo ""
echo "📋 Git Status:"
git status --short

# Ask for confirmation
echo ""
echo "Ready to commit and release v1.1.0?"
echo "Files to commit:"
git diff --name-only
git diff --cached --name-only
echo ""
echo "Proceed? (y/n)"
read -r response

if [ "$response" != "y" ]; then
    echo "Release cancelled"
    exit 0
fi

# Stage all changes
echo ""
echo "📦 Staging changes..."
git add -A

# Commit
echo ""
echo "💾 Committing..."
git commit -m "Release v1.1.0 - Module system & user-local architecture

Major features:
- Module system with 20+ pre-configured C++ libraries
- User-local architecture (~/.julia/replibuild/)
- Build system delegation (CMake, qmake, Make, Meson, Autotools, Cargo)
- Error learning with SQLite backend
- LLVM toolchain with automatic tool discovery
- 103/103 tests passing
- 20/20 modules working

Progression: v0.1.0 → v1.1.0 (major feature release)

Changes:
- Updated version to 1.1.0 in Project.toml and RepliBuild.jl
- Updated CHANGELOG.md to reflect v1.1.0
- Fixed README test counts (16→103)
- Clarified LLVM toolchain is production-ready
- Updated Julia compat to 1.9 (LTS support)
"

# Create tag
echo ""
echo "🏷️  Creating tag v1.1.0..."
git tag -a v1.1.0 -m "v1.1.0 - Module system & user-local architecture

Major features:
- Module system with 20+ pre-configured C++ libraries
- User-local architecture (~/.julia/replibuild/)
- Build system delegation
- Error learning with SQLite backend
- LLVM toolchain (production-ready)
- 103/103 tests passing

See CHANGELOG.md for full details.
"

# Show what will be pushed
echo ""
echo "📤 Ready to push:"
echo "   Branch: main"
echo "   Tag: v1.1.0"
echo ""
echo "Push to GitHub? (y/n)"
read -r response

if [ "$response" != "y" ]; then
    echo ""
    echo "⚠️  Changes committed locally but NOT pushed"
    echo "   To push later, run:"
    echo "   git push origin main"
    echo "   git push origin v1.1.0"
    exit 0
fi

# Push
echo ""
echo "⬆️  Pushing to GitHub..."
git push origin main
git push origin v1.1.0

# Success!
echo ""
echo "✅ Release v1.1.0 pushed successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 NEXT STEP: Register with Julia Registry"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Go to: https://github.com/obsidianjulua/RepliBuild.jl/commits/main"
echo "2. Click on the 'Release v1.1.0' commit"
echo "3. Add a comment: @JuliaRegistrator register"
echo "4. Submit the comment"
echo ""
echo "The bot will:"
echo "  ✓ Confirm registration (~30 seconds)"
echo "  ✓ Create PR to General registry (~1 minute)"
echo "  ✓ Run tests (~5 minutes)"
echo "  ✓ Auto-merge if tests pass (~15-30 minutes)"
echo ""
echo "Monitor at: https://github.com/JuliaRegistries/General/pulls"
echo ""
echo "🎉 Congratulations on the release!"
