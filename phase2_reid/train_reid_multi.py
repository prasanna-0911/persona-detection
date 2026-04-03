"""
Multi-Dataset Re-ID Training Script
=====================================
Trains the Re-ID model (ReIDNetwork — ResNet50 + 128-dim embedding) jointly on:
  - Market-1501
  - MSMT17
  - LaST

Training uses Batch-Hard Triplet Loss for efficient metric learning.
The combined dataset is concatenated and shuffled — every identity from every
dataset contributes to each epoch.

Usage:
    # Train from scratch (ImageNet pretrained backbone):
    python phase2_reid/train_reid_multi.py \\
        --market  path/to/Market-1501-v15.09.15 \\
        --msmt17  path/to/MSMT17_V1 \\
        --last    path/to/LaST \\
        --save-dir phase2_reid/checkpoints

    # Resume from checkpoint:
    python phase2_reid/train_reid_multi.py \\
        --market  path/to/Market-1501-v15.09.15 \\
        --msmt17  path/to/MSMT17_V1 \\
        --last    path/to/LaST \\
        --save-dir phase2_reid/checkpoints \\
        --resume  phase2_reid/checkpoints/latest_checkpoint.pth

    # Fine-tune from existing best_reid_model.pth:
    python phase2_reid/train_reid_multi.py \\
        --market  path/to/Market-1501-v15.09.15 \\
        --msmt17  path/to/MSMT17_V1 \\
        --last    path/to/LaST \\
        --save-dir phase2_reid/checkpoints \\
        --resume  phase2_reid/checkpoints/best_reid_model.pth \\
        --finetune   # lowers LR and unfreezes all layers

Requirements:
    pip install torch torchvision tqdm
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

# ── Make sure project root is on the path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'phase2_reid'))

from datasets.market1501 import Market1501Dataset
from datasets.msmt17 import MSMT17Dataset
from datasets.last import LaSTDataset
from models.reid_net import ReIDNetwork
from losses.triplet_loss import TripletLoss


# ─────────────────────────────────────────────────────────────────────────────
#  Default hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    'embedding_dim':  128,
    'batch_size':     32,
    'num_epochs':     40,
    'learning_rate':  1e-4,
    'finetune_lr':    5e-5,    # lower LR used when --finetune is passed
    'margin':         0.3,
    'num_workers':    4,
    'step_size':      15,      # LR scheduler step (epochs)
    'gamma':          0.5,     # LR decay factor
    'save_every':     5,       # save checkpoint every N epochs
}


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_dataset(args):
    """Concatenate all requested datasets into one ConcatDataset."""
    parts = []

    if args.market:
        print("[Dataset] Loading Market-1501 ...")
        parts.append(Market1501Dataset(root_dir=args.market, mode='train'))

    if args.msmt17:
        print("[Dataset] Loading MSMT17 ...")
        parts.append(MSMT17Dataset(root_dir=args.msmt17, mode='train'))

    if args.last:
        print("[Dataset] Loading LaST ...")
        parts.append(LaSTDataset(root_dir=args.last, mode='train'))

    if not parts:
        raise ValueError("At least one dataset path must be provided "
                         "(--market, --msmt17, --last).")

    combined = ConcatDataset(parts)
    print(f"\n[Dataset] Combined: {len(combined):,} total training samples "
          f"from {len(parts)} dataset(s)\n")
    return combined


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Run one training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)

    for batch_idx, (anchor, positive, negative, labels) in enumerate(pbar):
        anchor   = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()

        anchor_emb   = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        # Standard Triplet Loss computes distances natively using the 3 embeddings
        loss = criterion(anchor_emb, positive_emb, negative_emb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg': f'{avg:.4f}'})

    return running_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset Re-ID training on Market-1501 + MSMT17 + LaST.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset paths
    parser.add_argument('--market',   default=None,
                        help='Path to Market-1501-v15.09.15 root')
    parser.add_argument('--msmt17',   default=None,
                        help='Path to MSMT17_V1 root')
    parser.add_argument('--last',     default=None,
                        help='Path to LaST root')

    # Training config
    parser.add_argument('--save-dir', default='phase2_reid/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume',   default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--finetune', action='store_true',
                        help='Use lower LR (fine-tune mode)')
    parser.add_argument('--epochs',   type=int,   default=DEFAULTS['num_epochs'])
    parser.add_argument('--batch',    type=int,   default=DEFAULTS['batch_size'])
    parser.add_argument('--lr',       type=float, default=None,
                        help='Learning rate (overrides default)')
    parser.add_argument('--embedding-dim', type=int,
                        default=DEFAULTS['embedding_dim'])
    parser.add_argument('--margin',   type=float, default=DEFAULTS['margin'])
    parser.add_argument('--workers',  type=int,   default=DEFAULTS['num_workers'])
    parser.add_argument('--device',   default='cuda',
                        help="'cuda', 'cpu', or '0', '0,1' for multi-GPU")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        'cuda' if (args.device == 'cuda' and torch.cuda.is_available())
        else 'cpu'
    )
    print(f"\n{'='*60}")
    print(f"  Multi-Dataset Re-ID Training")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    combined = build_combined_dataset(args)
    loader = DataLoader(
        combined,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ReIDNetwork(embedding_dim=args.embedding_dim, pretrained=True)
    model = model.to(device)

    lr = args.lr or (DEFAULTS['finetune_lr'] if args.finetune
                     else DEFAULTS['learning_rate'])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=DEFAULTS['step_size'], gamma=DEFAULTS['gamma']
    )
    criterion = TripletLoss(margin=args.margin)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float('inf')
    history = []

    if args.resume and os.path.exists(args.resume):
        print(f"  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        if not args.finetune:
            # Only restore optimizer/scheduler state when not fine-tuning
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        history = ckpt.get('history', [])
        print(f"  Starting at epoch {start_epoch}, best loss so far: {best_loss:.4f}")

    if start_epoch > args.epochs:
        print("Training already complete!")
        return

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {total_params:,}")
    print(f"  Learning rate   : {lr}")
    print(f"  Batch size      : {args.batch}")
    print(f"  Epochs          : {start_epoch} → {args.epochs}")
    print(f"  Training samples: {len(combined):,}")
    print(f"\n{'='*60}\n")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, loader, criterion, optimizer, device, epoch, args.epochs
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch, 'loss': train_loss,
            'lr': current_lr, 'time_sec': round(elapsed, 1)
        })

        print(f"Epoch {epoch:>3}/{args.epochs}  |  "
              f"loss={train_loss:.4f}  |  "
              f"lr={current_lr:.2e}  |  "
              f"time={elapsed/60:.1f}min")

        # ── Save checkpoints ─────────────────────────────────────────────────
        ckpt_data = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss':                 train_loss,
            'best_loss':            best_loss,
            'history':              history,
            'embedding_dim':        args.embedding_dim,
        }

        # Always overwrite latest
        torch.save(ckpt_data, save_dir / 'latest_checkpoint.pth')

        # Best model
        if train_loss < best_loss:
            best_loss = train_loss
            ckpt_data['best_loss'] = best_loss
            torch.save(ckpt_data, save_dir / 'best_reid_model.pth')
            print(f"  ✔ New best model saved! (loss={best_loss:.4f})")

        # Periodic checkpoint
        if epoch % DEFAULTS['save_every'] == 0:
            torch.save(ckpt_data, save_dir / f'checkpoint_epoch_{epoch}.pth')

    # ── Save history ─────────────────────────────────────────────────────────
    hist_path = save_dir / 'training_history_multi.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best loss       : {best_loss:.4f}")
    print(f"  Best model      : {save_dir / 'best_reid_model.pth'}")
    print(f"  Training history: {hist_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
