#!/usr/bin/env python3
# Standalone: stack per-sequence ESM mean embeddings into one .pt for kNN reference DB.

import argparse
import csv
from pathlib import Path
import torch
from tqdm import tqdm

def detect_delimiter(path: Path, sample_bytes: int = 65536) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:sample_bytes]
    counts = {d: sample.count(d) for d in ["\t", ",", ";", "|"]}
    # default to tab if everything is zero
    delim = max(counts, key=counts.get)
    return delim if counts[delim] > 0 else "\t"

def extract_mean_rep(loaded_obj):
    """
    Supports ESM extract format:
      {'mean_representations': {layer: tensor[D]}, ...}
    Also accepts a raw tensor saved directly.
    """
    if torch.is_tensor(loaded_obj):
        v = loaded_obj
        return v.view(-1)

    if isinstance(loaded_obj, dict):
        if "mean_representations" in loaded_obj:
            rep_map = loaded_obj["mean_representations"]
            if 33 in rep_map:
                v = rep_map[33]
            else:
                # fallback: use last available layer
                last_layer = sorted(rep_map.keys())[-1]
                v = rep_map[last_layer]
            return v.view(-1)

        # some pipelines store under other keys; try common fallbacks
        for k in ["mean", "embedding", "emb", "repr", "representation"]:
            if k in loaded_obj and torch.is_tensor(loaded_obj[k]):
                return loaded_obj[k].view(-1)

    raise TypeError("Unsupported embedding object type/format in .pt file.")

def read_reference_ids_ecs(train_csv: Path, id_col: str, ec_col: str):
    delim = detect_delimiter(train_csv)
    ids, ecs = [], []
    with train_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {train_csv}")
        if id_col not in reader.fieldnames or ec_col not in reader.fieldnames:
            raise ValueError(
                f"Missing columns in {train_csv}. "
                f"Expected '{id_col}' and '{ec_col}'. Found={reader.fieldnames}"
            )
        for row in reader:
            pid = (row.get(id_col) or "").strip()
            ec  = (row.get(ec_col) or "").strip()
            if not pid or not ec:
                continue
            ids.append(pid)
            ecs.append(ec)
    if not ids:
        raise ValueError("No valid (Entry, EC) rows found.")
    return ids, ecs

def main():
    ap = argparse.ArgumentParser(
        description="Standalone: stack per-sequence ESM mean embeddings into a single .pt file."
    )
    ap.add_argument("--train_csv", required=True, help="Reference TSV/CSV with columns Entry and EC number.")
    ap.add_argument("--embeddings_dir", required=True, help="Directory containing <Entry>.pt embedding files.")
    ap.add_argument("--out_pt", required=True, help="Output stacked .pt file.")
    ap.add_argument("--id_col", default="Entry", help="ID column name (default: Entry).")
    ap.add_argument("--ec_col", default="EC number", help="EC column name (default: EC number).")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Output dtype.")
    ap.add_argument("--strict", action="store_true",
                    help="If set, missing embeddings raise error; otherwise missing rows are skipped with a warning.")
    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    emb_dir = Path(args.embeddings_dir)
    out_pt = Path(args.out_pt)

    ids, ecs = read_reference_ids_ecs(train_csv, args.id_col, args.ec_col)

    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

    kept_ids, kept_ecs, embs = [], [], []
    missing = 0

    for pid, ec in tqdm(list(zip(ids, ecs)), desc="Loading embeddings"):
        p = emb_dir / f"{pid}.pt"
        if not p.exists():
            missing += 1
            if args.strict:
                raise FileNotFoundError(f"Missing embedding file: {p}")
            continue

        obj = torch.load(p, map_location="cpu")
        v = extract_mean_rep(obj).to(dtype=out_dtype)

        if v.dim() != 1:
            v = v.view(-1)

        kept_ids.append(pid)
        kept_ecs.append(ec)
        embs.append(v)

    if not embs:
        raise RuntimeError("No embeddings were loaded. Check --embeddings_dir and Entry naming.")

    emb = torch.stack(embs, dim=0).contiguous()

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": kept_ids, "ec": kept_ecs, "esm": emb}, out_pt)

    print(f"Saved: {out_pt}")
    print(f"Shape: {tuple(emb.shape)}  dtype={emb.dtype}")
    if missing > 0:
        print(f"Warning: skipped {missing} entries without embeddings (use --strict to error).")

if __name__ == "__main__":
    main()
