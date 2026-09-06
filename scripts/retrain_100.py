"""Retrain predictor with expanded dataset - 100 epochs with progress bars."""
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import RenderPlanDataset
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from parameter_predictor import MultiHeadPredictor, RENDER_PLAN_KEYS

dataset_dir = r"D:\imgtocinamatic\CinematicAI\dataset"
checkpoint_dir = Path(r"D:\imgtocinamatic\CinematicAI\models\checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

train_ds = RenderPlanDataset(dataset_dir, split="train")
val_ds = RenderPlanDataset(dataset_dir, split="val")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = MultiHeadPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

print("\nStarting training (100 epochs)...")
best_val = float("inf")
start_time = time.time()

for epoch in range(1, 101):
    epoch_start = time.time()

    model.train()
    total_loss = 0
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/100 [train]", leave=False, ncols=80)
    for images, targets in train_pbar:
        images, targets = images.to(device), targets.to(device)
        pred, uncertainty = model(images)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train = total_loss / max(len(train_loader), 1)

    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:3d}/100 [val]  ", leave=False, ncols=80)
    with torch.no_grad():
        for images, targets in val_pbar:
            images, targets = images.to(device), targets.to(device)
            pred, uncertainty = model(images)
            val_loss += criterion(pred, targets).item()
            val_pbar.set_postfix(loss=f"{criterion(pred, targets).item():.4f}")
    avg_val = val_loss / max(len(val_loader), 1)
    scheduler.step()

    elapsed = time.time() - epoch_start
    improved = " *" if avg_val < best_val else ""

    if avg_val < best_val:
        best_val = avg_val
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val,
            "render_plan_keys": RENDER_PLAN_KEYS,
        }, checkpoint_dir / "best_predictor.pth")

    tqdm.write(f"  Epoch {epoch:3d}/100 | train={avg_train:.4f} | val={avg_val:.4f} | best={best_val:.4f} | {elapsed:.1f}s{improved}")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time/60:.1f} minutes")
print(f"Best val loss: {best_val:.6f}")
print(f"Checkpoint saved: {checkpoint_dir / 'best_predictor.pth'}")
