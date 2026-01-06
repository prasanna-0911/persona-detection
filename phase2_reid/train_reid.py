
"""
Training Script for Person Re-ID - WITH AUTO-SAVE EVERY EPOCH
"""

import os
import sys
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup paths
PROJECT_ROOT = '/content/drive/MyDrive/persona_detection_final'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f'{PROJECT_ROOT}/phase2_reid')

from datasets.market1501 import Market1501Dataset
from models.reid_net import ReIDNetwork
from losses.triplet_loss import TripletLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (anchor, positive, negative, _) in enumerate(pbar):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        
        optimizer.zero_grad()
        
        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)
        
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{running_loss / (batch_idx + 1):.4f}"
        })
    
    return running_loss / len(dataloader)


def main():
    # ==================== CONFIGURATION ====================
    CONFIG = {
        'data_root': '/content/drive/MyDrive/persona_detection_final/datasets/Market-1501-v15.09.15',
        'save_dir': '/content/drive/MyDrive/persona_detection_final/phase2_reid/checkpoints',
        'embedding_dim': 128,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0001,
        'margin': 0.3,
        'num_workers': 2,
    }
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== CHECK FOR RESUME ====================
    checkpoint_path = os.path.join(CONFIG['save_dir'], 'latest_checkpoint.pth')
    start_epoch = 1
    best_loss = float('inf')
    training_history = []
    
    if os.path.exists(checkpoint_path):
        print("\n🔄 RESUMING FROM CHECKPOINT...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        training_history = checkpoint.get('history', [])
        print(f"   Resuming from epoch {start_epoch}")
        print(f"   Best loss so far: {best_loss:.4f}")
    
    if start_epoch > CONFIG['num_epochs']:
        print("\n✅ Training already complete!")
        return
    
    # ==================== DATA ====================
    print("\n📂 Loading Dataset...")
    train_dataset = Market1501Dataset(
        root_dir=CONFIG['data_root'],
        mode='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # ==================== MODEL ====================
    print("\n🧠 Creating Model...")
    model = ReIDNetwork(
        embedding_dim=CONFIG['embedding_dim'],
        pretrained=True
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Load checkpoint if resuming
    if os.path.exists(checkpoint_path):
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    criterion = TripletLoss(margin=CONFIG['margin'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ==================== TRAINING LOOP ====================
    print("\n🚀 Starting Training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, CONFIG['num_epochs'] + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': current_lr
        })
        
        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # ===== SAVE CHECKPOINT EVERY EPOCH =====
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'best_loss': best_loss,
            'history': training_history,
            'config': CONFIG
        }
        
        # Always save latest
        torch.save(checkpoint_data, checkpoint_path)
        print(f"  💾 Saved checkpoint (epoch {epoch})")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = os.path.join(CONFIG['save_dir'], 'best_reid_model.pth')
            torch.save(checkpoint_data, best_path)
            print(f"  ✅ New best model saved!")
        
        # Save every 5 epochs
        if epoch % 5 == 0:
            epoch_path = os.path.join(CONFIG['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint_data, epoch_path)
    
    # ==================== TRAINING COMPLETE ====================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Model saved to: {CONFIG['save_dir']}")
    
    # Save final history
    history_path = os.path.join(CONFIG['save_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)


if __name__ == '__main__':
    main()
