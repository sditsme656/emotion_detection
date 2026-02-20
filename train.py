"""Training boilerplate for FER2013-based emotion classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import EMOTION_CLASSES, EmotionClassifier


def build_dataloaders(data_dir: Path, batch_size: int = 64):
    train_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_ds = datasets.ImageFolder(root=data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(root=data_dir / "val", transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)



def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)



def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_loader, val_loader = build_dataloaders(Path(args.data_dir), args.batch_size)
    model = EmotionClassifier(
        num_classes=len(EMOTION_CLASSES),
        backbone=args.backbone,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = out_dir / "best_emotion_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"[INFO] Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FER2013 emotion classifier")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root with train/ and val/")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
    )
    main(parser.parse_args())
