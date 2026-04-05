"""
Download and save barexam_qa (QA config) from HuggingFace.

Paper: "A Reasoning-Focused Legal Retrieval Benchmark" (Zheng et al., CS&Law 2025)
Task:  Multiple-choice bar exam QA; retrieve gold passage + select correct answer.
Size:  ~1,195 questions across train / validation / test splits.

All columns are flat strings — no nested structures.
Output: data/barexam_qa/barexam_qa.parquet  (single flat table with a 'split' column)
"""

import os
import pandas as pd
from huggingface_hub import hf_hub_download

OUT_DIR = "./data/barexam_qa"
os.makedirs(OUT_DIR, exist_ok=True)

splits = {}
for split in ("train", "validation", "test"):
    path = hf_hub_download(
        repo_id="reglab/barexam_qa",
        filename=f"data/qa/{split}.csv",
        repo_type="dataset",
        local_dir=OUT_DIR,
    )
    splits[split] = pd.read_csv(path).assign(split=split)
    print(f"  {split}: {len(splits[split])} rows")

df = pd.concat(splits.values(), ignore_index=True)

print(f"\nShape : {df.shape}")
print(f"Cols  : {df.columns.tolist()}")
print(df.head(2))

df.to_parquet(f"{OUT_DIR}/barexam_qa.parquet", index=False)
print(f"\nSaved → {OUT_DIR}/barexam_qa.parquet")
