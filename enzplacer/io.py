from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union

import pandas as pd
import torch


def read_reference_tsv(path: str, id_col: str = "Entry", ec_col: str = "EC number", seq_col: str = "Sequence") -> pd.DataFrame:
    """
    Load reference set (tab-delimited by default). Must contain at least: id_col, ec_col, seq_col.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    missing = [c for c in [id_col, ec_col, seq_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in reference TSV: {missing}. Found columns: {list(df.columns)}")
    return df[[id_col, ec_col, seq_col]].copy()


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    """
    Minimal FASTA reader.
    Returns (ids, sequences).
    """
    ids: List[str] = []
    seqs: List[str] = []
    cur_id: Optional[str] = None
    cur_seq: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur_seq))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            ids.append(cur_id)
            seqs.append("".join(cur_seq))
    return ids, seqs


def load_esm_mean_pt(path: str) -> torch.Tensor:
    """
    Load an ESM 'extract.py --include mean' artifact (.pt).
    Accepts either:
      - dict with ['mean_representations'][33]
      - a raw tensor (1280,)
    Returns float32 tensor shape (1280,).
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "mean_representations" in obj:
        v = obj["mean_representations"][33]
        return v.float()
    if torch.is_tensor(obj):
        return obj.float()
    raise ValueError(f"Unsupported embedding file format: {path}")


def load_embeddings_dict(path: str) -> Dict[str, Any]:
    """
    Load a stacked embeddings file saved as a dict with keys:
      - ids: List[str]
      - esm: Tensor[N, 1280]
    Optional keys: ec (List[str]), meta (dict)
    """
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Stacked embeddings must be a dict with keys {'ids','esm'}.")
    if "esm" not in obj and "emb" in obj:
        obj["esm"] = obj["emb"]
    if "ec" not in obj and "ecs" in obj:
        obj["ec"] = obj["ecs"]
    if "ids" not in obj or "esm" not in obj:
        raise ValueError("Stacked embeddings must be a dict with keys {'ids','esm'}.")
    if not torch.is_tensor(obj["esm"]):
        raise ValueError("Stacked embeddings 'esm' must be a torch.Tensor.")
    return obj
