#!/usr/bin/env bash
# Print the Decision 2.0 numbers from thai/out/d2.log (eval_all.sh): stage times, then per model the lines that matter.
cd "$(dirname "$0")/../out"
echo "--- stages"; grep -E '^====' d2.log
echo "--- errors"; grep -n -iE 'Traceback|Error|out of memory|Killed' d2.log | cut -c1-220 | head -20
echo "--- new questions"; grep -E '^== Decision|^  [0-9]+/12' d2.log | grep -B1 -E '^  [0-9]+/12'
echo "--- reviews (300 hand labels)"; grep -E 'hand-labelled reviews|^all +n=' d2.log
echo "--- real posts"; grep -E '^(long|short) +all' d2.log
echo "--- mood"; grep -E '^== .*: reviews|1-2 stars called|frustration separates|ambiguous / sarcastic|held-out composed' d2.log | cut -c1-420
echo "--- laya-thai-decisions, Thai suites"; grep -E '^== Decision-2.0-[A-Za-z0-9.-]+$| cases  acc ' d2.log | cut -c1-330
