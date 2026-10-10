#!/usr/bin/env bash
# Second half of the 2026-10-07 clean-up: the run folders are root-owned (written by the training container), so they are
# removed from a container; and the Decision 2.0 / Clef files live in the volume's shared blobs directory, so blobs that no
# remaining snapshot points to are removed (dry run first, then for real).
#   bash thai/cleanup_2026-10-07b.sh
set -uo pipefail
cd ~/laya/thai
echo "before: $(df -h / | awk 'NR==2{print $4" free of "$2}')"
echo "== A + B from a container (root)"
docker run --rm -v "$PWD/out":/out laya-train sh -c '
  for r in 3 4 5 6 7 8 9 10 11 12 13 15 17 18; do [ -d /out/laya-th-run$r ] && rm -rf /out/laya-th-run$r && echo "  removed run $r"; done
  for r in 14 16 19; do [ -d /out/laya-th-run$r/checkpoint_latest ] && rm -rf /out/laya-th-run$r/checkpoint_latest && echo "  removed run $r/checkpoint_latest"; done
  ls -d /out/laya-th-run* | tr "\n" " "; echo'
echo "== D. blobs nobody points to any more"
docker run --rm -v docker_hf-cache:/hf laya-train python - <<'EOF'
import os, sys
hub = "/hf/hub"
ref = set()
for name in os.listdir(hub):
    snap = os.path.join(hub, name, "snapshots")
    if not os.path.isdir(snap):
        continue
    for root, _, files in os.walk(snap):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                ref.add(os.path.basename(os.readlink(p)))
blobs = os.path.join(hub, "blobs")
gone, kept, freed = 0, 0, 0
for b in os.listdir(blobs):
    p = os.path.join(blobs, b)
    if b in ref:
        kept += 1
    else:
        freed += os.path.getsize(p); gone += 1; os.remove(p)
print(f"  referenced blobs kept: {kept}; unreferenced removed: {gone}, {freed / 2**30:.1f} GB")
EOF
docker run --rm -v docker_hf-cache:/hf laya-train sh -c 'du -sh /hf/hub/blobs'
echo "after:  $(df -h / | awk 'NR==2{print $4" free of "$2}')"
docker ps --format '{{.Names}} {{.Status}}' | head -3
