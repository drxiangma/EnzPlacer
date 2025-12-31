#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from enzplacer.io import read_reference_tsv, read_fasta, load_esm_mean_pt, load_embeddings_dict
from enzplacer.models import load_projector


def _l2_unit_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True).clamp(min=1e-12))


@torch.inference_mode()
def _project(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 4096, device: str = "cpu") -> torch.Tensor:
    outs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size].to(device)
        zb = model(xb)
        outs.append(zb.detach().cpu())
    return torch.cat(outs, dim=0)


def _vote(neigh_ecs: List[str], neigh_d: np.ndarray, vote: str) -> str:
    if vote == "nearest":
        return neigh_ecs[0]
    if vote == "uniform":
        w = np.ones_like(neigh_d, dtype=np.float64)
    elif vote == "inv_dist":
        w = 1.0 / np.clip(neigh_d, 1e-12, None)
    elif vote == "inv_dist_sq":
        w = 1.0 / np.clip(neigh_d, 1e-12, None) ** 2
    else:
        raise ValueError(f"Unknown vote: {vote}")

    scores: Dict[str, float] = {}
    for ec, ww in zip(neigh_ecs, w):
        scores[ec] = scores.get(ec, 0.0) + float(ww)
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _distinct_by_ec(neigh_ids: List[str], neigh_ecs: List[str], neigh_d: np.ndarray, k: int) -> Tuple[List[str], List[str], np.ndarray]:
    used = set()
    out_ids, out_ecs, out_d = [], [], []
    for pid, ec, dd in zip(neigh_ids, neigh_ecs, neigh_d):
        if ec in used:
            continue
        used.add(ec)
        out_ids.append(pid); out_ecs.append(ec); out_d.append(dd)
        if len(out_ids) >= k:
            break
    return out_ids, out_ecs, np.asarray(out_d, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="EnzPlacer sequence-level kNN EC inference (projector + reference embeddings).")

    # Inputs
    ap.add_argument("--train_data", default=None, help="Path to reference TSV/CSV (contains Entry, EC number, Sequence).")
    ap.add_argument("--test_fasta", required=True, help="Path to query FASTA.")
    ap.add_argument("--model", required=True, help="Path to projector checkpoint (.pth state_dict).")

    # Embeddings
    ap.add_argument("--reference_embeddings_pt", default=None,
                    help="Optional stacked reference embeddings (.pt dict keys: ids, ec, esm). If omitted, per-id .pt files are loaded from --embeddings_dir.")
    ap.add_argument("--query_embeddings_pt", default=None,
                    help="Optional stacked query embeddings (.pt dict keys: ids, esm). If omitted, per-id .pt files are loaded from --embeddings_dir using FASTA IDs.")
    ap.add_argument("--embeddings_dir", default="data/embeddings/esm_data",
                    help="Directory containing per-sequence ESM .pt files (<Entry>.pt) when stacked files are not provided.")

    # kNN
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--distance", choices=["l2","cos"], default="l2")
    ap.add_argument("--unit_norm_for_l2", action="store_true")
    ap.add_argument("--vote", choices=["nearest","uniform","inv_dist","inv_dist_sq"], default="nearest")
    ap.add_argument("--distinct_ecs", action="store_true")

    # Misc
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--out_csv", default="predictions.csv")
    ap.add_argument("--id_col", default="Entry")
    ap.add_argument("--ec_col", default="EC number")
    ap.add_argument("--seq_col", default="Sequence")
    args = ap.parse_args()

    # Load projector
    model, spec = load_projector(args.model, device=args.device)

    ref_ids: Optional[List[str]] = None
    ref_ecs: Optional[List[str]] = None
    if args.train_data:
        ref_df = read_reference_tsv(args.train_data, id_col=args.id_col, ec_col=args.ec_col, seq_col=args.seq_col)
        ref_ids = ref_df[args.id_col].astype(str).tolist()
        ref_ecs = ref_df[args.ec_col].astype(str).tolist()

    # Reference embeddings (ESM)
    if args.reference_embeddings_pt is not None:
        ref_obj = load_embeddings_dict(args.reference_embeddings_pt)
        X_ref = ref_obj["esm"].float()
        # Optional override of ids/ec from file
        if "ids" in ref_obj:
            ref_ids = [str(x) for x in ref_obj["ids"]]
        if "ec" in ref_obj:
            ref_ecs = [str(x) for x in ref_obj["ec"]]
    else:
        if not ref_ids:
            raise ValueError("Reference IDs are required when --reference_embeddings_pt is not provided.")
        embs = []
        for pid in tqdm(ref_ids, desc="Loading reference ESM embeddings"):
            embs.append(load_esm_mean_pt(f"{args.embeddings_dir}/{pid}.pt").unsqueeze(0))
        X_ref = torch.cat(embs, dim=0)
    if not ref_ids:
        raise ValueError("Reference IDs are required. Provide --train_data or ensure --reference_embeddings_pt contains 'ids'.")
    if not ref_ecs:
        raise ValueError("Reference EC labels are required. Provide --train_data or ensure --reference_embeddings_pt contains 'ec'.")

    # Queries
    q_ids, _q_seqs = read_fasta(args.test_fasta)
    if args.query_embeddings_pt is not None:
        q_obj = load_embeddings_dict(args.query_embeddings_pt)
        X_q = q_obj["esm"].float()
        if "ids" in q_obj:
            q_ids = [str(x) for x in q_obj["ids"]]
    else:
        embs = []
        for pid in tqdm(q_ids, desc="Loading query ESM embeddings"):
            embs.append(load_esm_mean_pt(f"{args.embeddings_dir}/{pid}.pt").unsqueeze(0))
        X_q = torch.cat(embs, dim=0)  # noqa: F821

    # Project to EnzPlacer embedding space
    Z_ref = _project(model, X_ref, batch_size=args.batch_size, device=args.device)
    Z_q = _project(model, X_q, batch_size=args.batch_size, device=args.device)

    # Distances
    if args.distance == "l2":
        if args.unit_norm_for_l2:
            Z_ref = _l2_unit_norm(Z_ref)
            Z_q = _l2_unit_norm(Z_q)
        # Compute pairwise squared L2 via (a-b)^2 = a^2 + b^2 - 2ab
        A = Z_q.numpy().astype(np.float64)
        B = Z_ref.numpy().astype(np.float64)
        a2 = (A * A).sum(axis=1, keepdims=True)  # (Q,1)
        b2 = (B * B).sum(axis=1, keepdims=True).T  # (1,R)
        D2 = a2 + b2 - 2.0 * (A @ B.T)
        D2 = np.maximum(D2, 0.0)
        D = np.sqrt(D2)
        smaller_is_better = True
    else:
        # cosine distance = 1 - cos sim
        A = Z_q.numpy().astype(np.float64)
        B = Z_ref.numpy().astype(np.float64)
        A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
        S = A @ B.T
        D = 1.0 - S
        smaller_is_better = True

    preds: List[str] = []
    for i in range(D.shape[0]):
        row = D[i]
        nn_idx = np.argsort(row)[: max(args.k, 1)]
        neigh_ids = [ref_ids[j] for j in nn_idx]
        neigh_ecs = [ref_ecs[j] for j in nn_idx]
        neigh_d = row[nn_idx].astype(np.float64)

        if args.distinct_ecs:
            neigh_ids, neigh_ecs, neigh_d = _distinct_by_ec(neigh_ids, neigh_ecs, neigh_d, k=args.k)
        else:
            neigh_ids, neigh_ecs, neigh_d = neigh_ids[:args.k], neigh_ecs[:args.k], neigh_d[:args.k]

        pred = _vote(neigh_ecs, neigh_d, vote=args.vote)
        preds.append(pred)

    out = pd.DataFrame({"id": q_ids, "pred_ec": preds})
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} | n={len(out)}")


if __name__ == "__main__":
    main()
