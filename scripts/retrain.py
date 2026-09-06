"""Quick retrain with expanded dataset."""
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import RenderPlanDataset
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
from parameter_predictor import MultiHeadPredictor, RENDER_PLAN_KEYS

dataset_dir = r"D:\imgtocinamatic\CinematicAI\dataset"
checkpoint_dir = Path(r"D:\imgtocinamatic\CinematicAI\models\checkpoints")

train_ds = RenderPlanDataset(dataset_dir, split="train")
val_ds = RenderPlanDataset(dataset_dir, split="val")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = MultiHeadPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.MSELoss()

print("Starting training...")
best_val = float("inf")
for epoch in range(20):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        pred, uncertainty = model(images)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train = total_loss / max(len(train_loader), 1)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            pred, uncertainty = model(images)
            val_loss += criterion(pred, targets).item()
    avg_val = val_loss / max(len(val_loader), 1)
    scheduler.step()

    if avg_val < best_val:
        best_val = avg_val
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "val_loss": avg_val,
            "render_plan_keys": RENDER_PLAN_KEYS,
        }, checkpoint_dir / "best_predictor.pth")

    print(f"Epoch {epoch+1}/20: train={avg_train:.4f}, val={avg_val:.4f}, best={best_val:.4f}")

print("Training complete!")
print(f"Best val loss: {best_val:.4f}")
