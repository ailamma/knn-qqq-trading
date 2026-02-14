#!/bin/bash
# KNN QQQ Trading Model — Environment Bootstrap
# Run this at the start of every session to set up the environment.

set -e

echo "=== KNN QQQ Trading Model — Session Bootstrap ==="
echo ""

# 1. Check Python version
echo "1. Python version:"
python3 --version
echo ""

# 2. Install dependencies if needed
if [ ! -f ".deps_installed" ]; then
    echo "2. Installing dependencies..."
    pip install -r requirements.txt --break-system-packages -q 2>/dev/null
    if [ $? -eq 0 ]; then
        touch .deps_installed
        echo "   Dependencies installed."
    else
        echo "   WARNING: pip install failed (no network?). Install manually:"
        echo "   pip install -r requirements.txt --break-system-packages"
    fi
else
    echo "2. Dependencies already installed."
fi
echo ""

# 3. Show project structure
echo "3. Project structure:"
find . -type f -not -path './.git/*' -not -path './.deps_installed' | head -40
echo ""

# 4. Show git status
echo "4. Git status:"
git log --oneline -10 2>/dev/null || echo "   No git history yet."
echo ""

# 5. Show current progress
echo "5. Current progress:"
if [ -f "claude-progress.txt" ]; then
    tail -20 claude-progress.txt
else
    echo "   No progress file found."
fi
echo ""

# 6. Show feature completion status
echo "6. Feature completion:"
if [ -f "feature_list.json" ]; then
    total=$(python3 -c "import json; data=json.load(open('feature_list.json')); print(sum(len(p['features']) for p in data['phases']))")
    done=$(python3 -c "import json; data=json.load(open('feature_list.json')); print(sum(1 for p in data['phases'] for f in p['features'] if f['passes']))")
    echo "   Completed: $done / $total features"
else
    echo "   No feature list found."
fi
echo ""

# 7. Quick data check
echo "7. Data status:"
if [ -d "data/raw" ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "   Raw data files:"
    ls -la data/raw/*.csv 2>/dev/null || echo "   No CSV files yet."
else
    echo "   No raw data downloaded yet."
fi
if [ -d "data/processed" ] && [ "$(ls -A data/processed 2>/dev/null)" ]; then
    echo "   Processed data files:"
    ls -la data/processed/*.csv 2>/dev/null || echo "   No processed files yet."
else
    echo "   No processed data yet."
fi
echo ""

echo "=== Bootstrap complete. Ready to work. ==="
