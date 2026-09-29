"""Look at Multimodal-Mind2Web without the screenshots: parquet schema + the text columns of two rows (range reads over hf://)."""
import json

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO = "osunlp/Multimodal-Mind2Web"
fs = HfFileSystem()
files = sorted(f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", recursive=True) if f.path.endswith(".parquet"))
by_split = {}
for f in files:
    by_split.setdefault(f.split("/")[-1].split("-")[0], []).append(f)
print({k: len(v) for k, v in by_split.items()})
path = f"datasets/{REPO}/{files[0]}"
pf = pq.ParquetFile(path, filesystem=fs)
print(pf.schema_arrow)
print("rows in first file", pf.metadata.num_rows)
text_cols = [n for n in pf.schema_arrow.names if n not in ("screenshot", "raw_html")]
tab = pf.read_row_group(0, columns=text_cols).slice(0, 2).to_pylist()
for row in tab:
    print("=" * 80)
    for k, v in row.items():
        s = v if isinstance(v, (int, float)) else json.dumps(v, ensure_ascii=False)
        print(f"--- {k} ({len(s) if isinstance(s, str) else ''}): {s[:700] if isinstance(s, str) else s}")
