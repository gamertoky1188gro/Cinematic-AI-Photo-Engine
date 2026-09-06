import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms.functional import to_tensor, resize

import sys

_models_dir = Path(__file__).resolve().parent / "models"
if str(_models_dir) not in sys.path:
    sys.path.insert(0, str(_models_dir))

from parameter_predictor import MultiHeadPredictor, RENDER_PLAN_KEYS, count_parameters


class RenderPlanDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str = "train"):
        self.dataset_dir = Path(dataset_dir)
        with open(self.dataset_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.samples = [
            {
                "image_path": self.dataset_dir / "images" / e["image"],
                "plan": e["render_plan"],
            }
            for e in self.metadata["images"]
            if (self.dataset_dir / "images" / e["image"]).exists()
        ]

        n = len(self.samples)
        if n == 0:
            raise ValueError(f"No valid samples found in {dataset_dir}")

        split_idx = max(1, int(n * 0.8))
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = resize(image, (224, 224))
        image = to_tensor(image)

        targets = torch.tensor(
            [sample["plan"][k] for k in RENDER_PLAN_KEYS],
            dtype=torch.float32,
        )

        return image, targets


def train(
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 4,
    device_str: str | None = None,
):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    train_ds = RenderPlanDataset(dataset_dir, split="train")
    val_ds = RenderPlanDataset(dataset_dir, split="val")

    if len(val_ds) == 0:
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultiHeadPredictor(pretrained=True)
    model.freeze_backbone()
    model.to(device)

    print(f"Parameters: {count_parameters(model)}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            pred_params, pred_uncert = model(images)
            loss = criterion(pred_params, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                pred_params, _ = model(images)
                val_loss += criterion(pred_params, targets).item() * images.size(0)
        val_loss /= max(len(val_ds), 1)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "render_plan_keys": RENDER_PLAN_KEYS,
                },
                output_dir / "best_predictor.pth",
            )

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_loss": best_val_loss,
            "render_plan_keys": RENDER_PLAN_KEYS,
        },
        output_dir / "final_predictor.pth",
    )

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved: {output_dir / 'best_predictor.pth'}")

    return {"best_val_loss": best_val_loss, "epochs": epochs}


def main():
    parser = argparse.ArgumentParser(description="Train parameter predictor")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output", type=str, default="models/checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
