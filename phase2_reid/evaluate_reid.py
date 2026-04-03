"""
Re-ID Model Evaluation Script
================================
Evaluates a trained Re-ID model using standard benchmarks:
  - Rank-1, Rank-5, Rank-10 accuracy (CMC curve)
  - mean Average Precision (mAP)

Supports evaluation on:
  - Market-1501 (query vs gallery)
  - MSMT17      (query vs gallery)
  - LaST        (val/query vs val/gallery)

Usage:
    # Evaluate on Market-1501:
    python phase2_reid/evaluate_reid.py \\
        --model   phase2_reid/checkpoints/best_reid_model.pth \\
        --dataset market1501 \\
        --root    path/to/Market-1501-v15.09.15

    # Evaluate on MSMT17:
    python phase2_reid/evaluate_reid.py \\
        --model   phase2_reid/checkpoints/best_reid_model.pth \\
        --dataset msmt17 \\
        --root    path/to/MSMT17_V1

    # Evaluate on LaST (val split):
    python phase2_reid/evaluate_reid.py \\
        --model   phase2_reid/checkpoints/best_reid_model.pth \\
        --dataset last \\
        --root    path/to/LaST

Requirements:
    pip install torch torchvision tqdm
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase2_reid'))

from models.reid_net import ReIDNetwork
from datasets.market1501 import Market1501Query, Market1501Gallery
from datasets.msmt17 import MSMT17Query, MSMT17Gallery
from datasets.last import LaSTSplit


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(model, loader, device):
    """
    Run inference on a query or gallery DataLoader.

    Returns:
        features   : np.ndarray [N, embedding_dim]
        person_ids : np.ndarray [N]
        camera_ids : np.ndarray [N]
    """
    model.eval()
    all_features = []
    all_pids = []
    all_cids = []

    with torch.no_grad():
        for imgs, pids, cids in tqdm(loader, desc='  Extracting', leave=False):
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            all_features.append(feats)
            all_pids.extend(pids.numpy() if hasattr(pids, 'numpy') else list(pids))
            all_cids.extend(cids.numpy() if hasattr(cids, 'numpy') else list(cids))

    if not all_features:
        raise RuntimeError(
            "No images were loaded from the dataset! "
            "Please check if the dataset path is correct and the images are extracted properly."
        )

    return (
        np.concatenate(all_features, axis=0),
        np.array(all_pids),
        np.array(all_cids),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CMC + mAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_cmc_map(query_feats, query_pids, query_cids,
                    gallery_feats, gallery_pids, gallery_cids,
                    max_rank=10):
    """
    Compute CMC curve and mAP.

    For each query image:
      1. Compute cosine similarity to all gallery images.
      2. Sort gallery by descending similarity.
      3. Remove same-camera same-identity images (standard Re-ID protocol).
      4. Check if correct match appears at rank 1, 5, 10.
      5. Compute Average Precision for this query.

    Returns:
        cmc  : np.ndarray [max_rank]  — cumulative match characteristic
        mAP  : float                  — mean average precision
    """
    num_query = query_feats.shape[0]
    cmc_scores = np.zeros(max_rank, dtype=float)
    ap_list = []

    # Normalize features for cosine similarity
    q_norm = query_feats / (np.linalg.norm(query_feats, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_feats / (np.linalg.norm(gallery_feats, axis=1, keepdims=True) + 1e-8)

    # Similarity matrix [Q, G]
    sim_matrix = np.dot(q_norm, g_norm.T)

    for q_idx in range(num_query):
        q_pid = query_pids[q_idx]
        q_cid = query_cids[q_idx]

        sims = sim_matrix[q_idx]           # [G]
        order = np.argsort(-sims)          # descending

        # Filter: remove same camera + same person (query-to-junk removal)
        keep = []
        for g_idx in order:
            g_pid = gallery_pids[g_idx]
            g_cid = gallery_cids[g_idx]

            # Skip if same camera AND same person (junk match)
            if g_pid == q_pid and g_cid == q_cid:
                continue
            keep.append(g_idx)

        if not keep:
            continue

        matches = (gallery_pids[np.array(keep)] == q_pid).astype(int)

        # CMC: rank-k hit
        for k in range(min(max_rank, len(matches))):
            if matches[k] == 1:
                cmc_scores[k:] += 1
                break

        # AP: area under precision-recall curve
        num_relevant = matches.sum()
        if num_relevant == 0:
            ap_list.append(0.0)
            continue

        precision_at_k = 0.0
        hits = 0
        for k, is_match in enumerate(matches, 1):
            if is_match:
                hits += 1
                precision_at_k += hits / k
        ap = precision_at_k / num_relevant
        ap_list.append(ap)

    cmc = cmc_scores / num_query
    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    return cmc, mAP


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def get_query_gallery(dataset_name: str, root: str, batch: int, workers: int):
    """Return (query_loader, gallery_loader) for the chosen dataset."""

    if dataset_name == 'market1501':
        query   = Market1501Query(root_dir=root)
        gallery = Market1501Gallery(root_dir=root)

    elif dataset_name == 'msmt17':
        query   = MSMT17Query(root_dir=root)
        gallery = MSMT17Gallery(root_dir=root)

    elif dataset_name == 'last':
        query   = LaSTSplit(root_dir=root, split='test/query')
        gallery = LaSTSplit(root_dir=root, split='test/gallery')

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Choose from: market1501, msmt17, last")

    q_loader = DataLoader(query,   batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)
    g_loader = DataLoader(gallery, batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=True)
    return q_loader, g_loader


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Re-ID model — CMC + mAP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model',   required=True,
                        help='Path to trained checkpoint (.pth)')
    parser.add_argument('--dataset', required=True,
                        choices=['market1501', 'msmt17', 'last'],
                        help='Evaluation dataset')
    parser.add_argument('--root',    required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--output',  default=None,
                        help='Path to save JSON report (optional)')
    parser.add_argument('--batch',   type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-rank', type=int, default=10,
                        help='Max rank for CMC curve (default: 10)')
    parser.add_argument('--device',  default='cuda')
    args = parser.parse_args()

    device = torch.device(
        'cuda' if (args.device == 'cuda' and torch.cuda.is_available())
        else 'cpu'
    )

    print(f"\n{'='*60}")
    print(f"  Re-ID Evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model  : {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.model, map_location=device)
    embedding_dim = ckpt.get('embedding_dim', 128)
    model = ReIDNetwork(embedding_dim=embedding_dim, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded model | embedding_dim={embedding_dim} | "
          f"trained epochs={ckpt.get('epoch', '?')}")

    # ── Load data ─────────────────────────────────────────────────────────────
    q_loader, g_loader = get_query_gallery(
        args.dataset, args.root, args.batch, args.workers
    )

    # ── Extract features ──────────────────────────────────────────────────────
    print("\n[1/3] Extracting query features ...")
    q_feats, q_pids, q_cids = extract_features(model, q_loader, device)

    print("[2/3] Extracting gallery features ...")
    g_feats, g_pids, g_cids = extract_features(model, g_loader, device)

    print(f"      Query  : {q_feats.shape[0]:,} images")
    print(f"      Gallery: {g_feats.shape[0]:,} images")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("[3/3] Computing CMC + mAP ...")
    cmc, mAP = compute_cmc_map(
        q_feats, q_pids, q_cids,
        g_feats, g_pids, g_cids,
        max_rank=args.max_rank,
    )

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results on {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  mAP     : {mAP*100:.2f}%")
    for k in [1, 5, 10]:
        if k <= args.max_rank:
            print(f"  Rank-{k:<2}  : {cmc[k-1]*100:.2f}%")
    print(f"{'='*60}\n")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        'model':     args.model,
        'dataset':   args.dataset,
        'timestamp': datetime.now().isoformat(),
        'mAP':       round(mAP * 100, 3),
        'rank_1':    round(cmc[0] * 100, 3),
        'rank_5':    round(cmc[4] * 100, 3) if args.max_rank >= 5 else None,
        'rank_10':   round(cmc[9] * 100, 3) if args.max_rank >= 10 else None,
        'cmc_curve': [round(v * 100, 3) for v in cmc.tolist()],
    }

    out_path = args.output or f'eval_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M")}.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved → {out_path}")


if __name__ == '__main__':
    main()
