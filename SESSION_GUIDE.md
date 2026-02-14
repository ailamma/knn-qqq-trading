# Session Guide — How to Resume Work

This file tells you (Claude or human) exactly how to resume work on this project
from any new context window or coding session.

## Step 1: Orient yourself
```bash
cd knn-qqq-trading
bash init.sh
```
This shows you: Python version, installed deps, project files, git history,
current progress, and feature completion count.

## Step 2: Read the progress log
```bash
cat claude-progress.txt
```
Look at the **last session entry** to understand:
- What was done last
- What the next step should be
- Any blockers or notes

## Step 3: Check the feature list
```bash
python3 -c "
import json
data = json.load(open('feature_list.json'))
for phase in data['phases']:
    done = sum(1 for f in phase['features'] if f['passes'])
    total = len(phase['features'])
    status = '✅' if done == total else '🔴' if done == 0 else '🟡'
    print(f\"{status} Phase {phase['phase']}: {phase['name']} ({done}/{total})\"  )
    for f in phase['features']:
        mark = '✅' if f['passes'] else '⬜'
        print(f\"   {mark} {f['id']}: {f['description']}\")
"
```

## Step 4: Pick the next feature
Choose the **highest-priority feature** that has `passes: false`.
Work on **only one feature** per session.

## Step 5: Implement the feature
Follow the `steps` listed in feature_list.json for that feature.

## Step 6: Before ending the session

### a) Commit your code
```bash
git add -A
git commit -m "Complete [FEATURE_ID]: [description]

- [what was implemented]
- [key decisions made]
- [test results if applicable]"
```

### b) Update feature_list.json
Change the feature's `passes` field from `false` to `true`.

### c) Update claude-progress.txt
Add a new session entry at the bottom:
```
### Session N — [Date]
- **What was done:** [summary]
- **Current state:** [what works now]
- **Next step:** [which feature to tackle next]
- **Blockers:** [any issues]
- **Features completed:** X / 28
```

### d) Final commit
```bash
git add -A
git commit -m "Session N complete: [FEATURE_ID] done, updated progress"
```

## Rules
1. **Never** edit feature descriptions or steps in feature_list.json — only flip `passes` to `true`
2. **Never** use TQQQ/SQQQ data in model training (Phases 1-4)
3. **Always** leave code in a runnable state — no half-finished functions
4. **Always** commit before ending a session
5. **One feature per session** — don't try to do too much
