# EnzPlacer

EnzPlacer is a contrastive learning approach for enzyme function prediction from sequence, introduced in
"How Not to be Seen: Predicting Unseen Enzyme Functions using Contrastive Learning." The key challenge we address
is that many enzymes in newly sequenced genomes have functions not yet characterized, so their exact EC labels
do not exist at training time. Rather than forcing a wrong label, EnzPlacer places sequences into the known
functional landscape as accurately as possible, producing testable hypotheses for experimental follow-up.

Our model predicts the 1st, 2nd, and 3rd EC numbers for proteins whose 4th EC number was unseen during training,
providing a narrowed and biologically meaningful functional context even when the precise function is unknown.

This repository provides an inference pipeline for the EnzPlacer model,
which maps ESM mean embeddings (1280-d) into the EnzPlacer embedding space. The reference database is then
used for the nearest-neighbor label transfer.

## 1) Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Quickstart (mock data)

The repo contains a small mock reference TSV, a mock query FASTA, and mock precomputed embeddings.

```bash
python scripts/infer_knn.py \
  --train_data data/reference/mock_reference.tsv \
  --test_fasta data/query/mock_query.fasta \
  --model checkpoints/EnzPlacer.pth \
  --reference_embeddings_pt data/reference/mock_reference_embeddings.pt \
  --query_embeddings_pt data/query/mock_query_embeddings.pt \
  --distance l2 --unit_norm_for_l2 \
  --k 7 --vote nearest --distinct_ecs \
  --out_csv predictions.csv
```

Output: `predictions.csv` with columns `id, pred_ec`.

## 3) Prepare query FASTA + ESM embeddings (real data)

If you have a query TSV/CSV with columns `Entry`, `Sequence` (EC number optional), you can generate a FASTA
and the stacked query embeddings in one step (ESM mean embeddings):

```bash
python scripts/prepare_query_embeddings.py \
  --test_tsv data/query/my_query.tsv \
  --out_fasta data/query/my_query.fasta \
  --out_embeddings_pt data/query/my_query_embeddings.pt
```

If you already have a query FASTA, generate the embeddings directly:

```bash
python scripts/prepare_query_embeddings.py \
  --test_fasta data/query/my_query.fasta \
  --out_embeddings_pt data/query/my_query_embeddings.pt
```

Optional: add `--embeddings_dir data/embeddings/esm_data` to also write per-sequence `<Entry>.pt` files.

Note: this script uses the ESM-1b model and may download weights on first run.

## 4) Inference (real data)

Download the reference CSV and precomputed embeddings from https://doi.org/10.5281/zenodo.18110452
and place them at:

- `data/reference/train_unseen.csv`
- `data/reference/train_unseen_embeddings.pt`

Then run inference (the downloaded reference embeddings already include IDs and EC labels, so `--train_data` is optional):

```bash
python scripts/infer_knn.py \
  --train_data data/reference/train_unseen.csv \
  --test_fasta data/query/my_query.fasta \
  --model checkpoints/EnzPlacer.pth \
  --reference_embeddings_pt data/reference/train_unseen_embeddings.pt \
  --query_embeddings_pt data/query/my_query_embeddings.pt \
  --distance l2 --unit_norm_for_l2 \
  --k 7 --vote nearest --distinct_ecs \
  --out_csv preds_my_query.csv
```

If you omit `--train_data`, the reference embeddings file must include `ids` and `ec`.
If `--query_embeddings_pt` is not provided, query embeddings are loaded from `--embeddings_dir` using FASTA IDs.

## 5) License

GPL-3.0-only (see `LICENSE`).
