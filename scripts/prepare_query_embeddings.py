#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
from tqdm import tqdm

from enzplacer.io import read_fasta


def read_query_tsv(path: Path, id_col: str, seq_col: str) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(path, sep=None, engine="python")
    missing = [c for c in [id_col, seq_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in query TSV/CSV: {missing}. Found: {list(df.columns)}")
    ids = df[id_col].astype(str).tolist()
    seqs = df[seq_col].astype(str).tolist()
    if not ids or not seqs:
        raise ValueError("No query sequences found.")
    return ids, seqs


def write_fasta(path: Path, ids: List[str], seqs: List[str], line_len: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for pid, seq in zip(ids, seqs):
            f.write(f">{pid}\n")
            for i in range(0, len(seq), line_len):
                f.write(seq[i:i + line_len] + "\n")

def _handle_long_sequences(
    ids: List[str],
    seqs: List[str],
    max_len: int,
    mode: str,
) -> Tuple[List[str], List[str]]:
    if mode == "error":
        for pid, seq in zip(ids, seqs):
            if len(seq) > max_len:
                raise ValueError(f"Sequence length {len(seq)} above maximum {max_len} for Entry={pid}.")
        return ids, seqs

    out_ids: List[str] = []
    out_seqs: List[str] = []
    for pid, seq in zip(ids, seqs):
        if len(seq) <= max_len:
            out_ids.append(pid)
            out_seqs.append(seq)
            continue
        if mode == "skip":
            print(f"Warning: skipping {pid} (length {len(seq)} > {max_len}).", file=sys.stderr)
            continue
        if mode == "truncate":
            # print(f"Warning: truncating {pid} from {len(seq)} to {max_len}.", file=sys.stderr)
            out_ids.append(pid)
            out_seqs.append(seq[:max_len])
            continue
        raise ValueError(f"Unknown long_seq mode: {mode}")
    if not out_ids:
        raise ValueError("No sequences left after applying long-sequence handling.")
    return out_ids, out_seqs


@torch.inference_mode()
def compute_esm1b_mean_embeddings(
    ids: List[str],
    seqs: List[str],
    device: str,
    max_tokens: int,
) -> torch.Tensor:
    import esm

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    items = list(enumerate(zip(ids, seqs)))
    items.sort(key=lambda x: len(x[1][1]), reverse=True)

    out = [None] * len(items)
    current = []
    current_tokens = 0

    pbar = tqdm(total=len(items), desc="ESM-1b embedding", unit="seq")

    def flush(batch_items):
        if not batch_items:
            return
        batch = [(pid, seq) for _, (pid, seq) in batch_items]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        res = model(tokens, repr_layers=[33], return_contacts=False)
        reps = res["representations"][33]
        for i, (orig_idx, (pid, seq)) in enumerate(batch_items):
            seq_len = (tokens[i] != alphabet.padding_idx).sum().item()
            mean_rep = reps[i, 1:seq_len - 1].mean(0).detach().cpu()
            out[orig_idx] = mean_rep
        pbar.update(len(batch_items))

    for item in items:
        _, (_, seq) = item
        seq_tokens = len(seq) + 2
        if current and current_tokens + seq_tokens > max_tokens:
            flush(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += seq_tokens
    flush(current)
    pbar.close()

    embs = []
    for v in out:
        if v is None:
            raise RuntimeError("Failed to compute one or more embeddings.")
        embs.append(v)
    return torch.stack(embs, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate query FASTA and ESM-1b mean embeddings (stacked .pt) for inference."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--test_tsv", help="Query TSV/CSV with columns Entry and Sequence (and optional EC number).")
    group.add_argument("--test_fasta", help="Query FASTA.")

    ap.add_argument("--out_fasta", default=None, help="Output FASTA path when --test_tsv is used.")
    ap.add_argument("--out_embeddings_pt", default=None, help="Output stacked embeddings .pt.")
    ap.add_argument("--embeddings_dir", default=None, help="Optional directory to write per-sequence <Entry>.pt files.")
    ap.add_argument("--id_col", default="Entry")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--max_tokens", type=int, default=4096, help="Max tokens per batch (default: 4096).")
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Max token length for ESM-1b including special tokens (default: 1024).",
    )
    ap.add_argument("--long_seq", choices=["error", "skip", "truncate"], default="truncate",
                    help="How to handle sequences longer than --max_seq_len.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ids: List[str]
    seqs: List[str]
    out_fasta: Optional[Path] = None
    source_path = Path(args.test_tsv or args.test_fasta)

    if args.test_tsv:
        ids, seqs = read_query_tsv(source_path, args.id_col, args.seq_col)
        if args.out_fasta:
            out_fasta = Path(args.out_fasta)
        else:
            out_fasta = source_path.with_suffix(".fasta")
        write_fasta(out_fasta, ids, seqs)
        print(f"Saved: {out_fasta}")
    else:
        ids, seqs = read_fasta(str(source_path))

    max_res_len = max(args.max_seq_len - 2, 1)
    ids, seqs = _handle_long_sequences(ids, seqs, max_len=max_res_len, mode=args.long_seq)

    if args.out_embeddings_pt:
        out_embeddings_pt = Path(args.out_embeddings_pt)
    else:
        out_embeddings_pt = source_path.parent / f"{source_path.stem}_embeddings.pt"

    embs = compute_esm1b_mean_embeddings(ids, seqs, device=args.device, max_tokens=args.max_tokens)
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    embs = embs.to(dtype=out_dtype)

    out_embeddings_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "esm": embs}, out_embeddings_pt)
    print(f"Saved: {out_embeddings_pt}")

    if args.embeddings_dir:
        emb_dir = Path(args.embeddings_dir)
        emb_dir.mkdir(parents=True, exist_ok=True)
        for pid, v in zip(ids, embs):
            obj = {"mean_representations": {33: v}}
            torch.save(obj, emb_dir / f"{pid}.pt")
        print(f"Saved per-sequence .pt files in: {emb_dir}")


if __name__ == "__main__":
    main()
