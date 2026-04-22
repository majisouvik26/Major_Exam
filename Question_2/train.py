import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from dataloader import create_dataloaders
from model import UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for 23-class segmentation")
    parser.add_argument("--data_root", type=str, default="MLDLOPs_2026_Major_Exam")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_width", type=int, default=128)
    parser.add_argument("--image_height", type=int, default=96)
    parser.add_argument("--num_classes", type=int, default=23)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="major-exam-segmentation")
    parser.add_argument("--wandb_run_name", type=str, default="unet-23class")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _update_confusion_matrix(conf_mat, preds, targets, num_classes):
    valid = (targets >= 0) & (targets < num_classes)
    flat_targets = targets[valid]
    flat_preds = preds[valid]
    bincount = torch.bincount(
        num_classes * flat_targets + flat_preds,
        minlength=num_classes ** 2,
    )
    conf_mat += bincount.reshape(num_classes, num_classes)


def compute_metrics_from_confmat(conf_mat, eps=1e-7):
    tp = conf_mat.diag().float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp

    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    return iou.mean().item(), dice.mean().item()


def run_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    total_samples = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    progress = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        _update_confusion_matrix(conf_mat, preds, masks, num_classes)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = running_loss / max(total_samples, 1)
    miou, mdice = compute_metrics_from_confmat(conf_mat)
    return mean_loss, miou, mdice


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = torch.argmax(logits, dim=1)
        _update_confusion_matrix(conf_mat, preds, masks, num_classes)

    mean_loss = running_loss / max(total_samples, 1)
    miou, mdice = compute_metrics_from_confmat(conf_mat)
    return mean_loss, miou, mdice


@torch.no_grad()
def evaluate_test_set(model, loader, device, num_classes):
    model.eval()
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        _update_confusion_matrix(conf_mat, preds, masks, num_classes)

    miou, mdice = compute_metrics_from_confmat(conf_mat)
    return miou, mdice


def save_plots(history, output_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["test_loss"], marker="o", label="Test")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_miou"], marker="o", label="Train")
    plt.plot(epochs, history["test_miou"], marker="o", label="Test")
    plt.title("mIoU")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_mdice"], marker="o", label="Train")
    plt.plot(epochs, history["test_mdice"], marker="o", label="Test")
    plt.title("mDice")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=180)
    plt.close()


def main():
    args = parse_args()
    if not args.eval_only and args.epochs < 15:
        raise ValueError("Please train for at least 15 epochs as required.")

    set_seed(args.seed)
    print("Starting run...", flush=True)
    print(f"Data root: {args.data_root}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}", flush=True)

    image_size = (args.image_width, args.image_height)
    print("Preparing dataloaders...", flush=True)
    train_loader, test_loader, _ = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        train_ratio=0.8,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(
        f"Dataloaders ready | train_batches={len(train_loader)} test_batches={len(test_loader)}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model = UNet(in_channels=3, num_classes=args.num_classes).to(device)

    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint when using --eval_only.")

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        test_miou, test_mdice = evaluate_test_set(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=args.num_classes,
        )

        print(f"mIOU: {test_miou:.4f}")
        print(f"mDICE: {test_mdice:.4f}")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = {
        "train_loss": [],
        "train_miou": [],
        "train_mdice": [],
        "test_loss": [],
        "test_miou": [],
        "test_mdice": [],
    }

    run_wandb = False
    wandb = None
    if args.use_wandb:
        try:
            import wandb as wandb_lib

            wandb = wandb_lib
            print(f"Initializing wandb ({args.wandb_mode})...", flush=True)
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                mode=args.wandb_mode,
            )
            run_wandb = True
            print("wandb initialized", flush=True)
        except Exception as exc:
            print(f"wandb disabled: {exc}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou, train_mdice = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_classes=args.num_classes,
        )

        test_loss, test_miou, test_mdice = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["train_mdice"].append(train_mdice)
        history["test_loss"].append(test_loss)
        history["test_miou"].append(test_miou)
        history["test_mdice"].append(test_mdice)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_mIoU={train_miou:.4f} train_mDice={train_mdice:.4f} | "
            f"test_loss={test_loss:.4f} test_mIoU={test_miou:.4f} test_mDice={test_mdice:.4f}"
        )

        if run_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/mIoU": train_miou,
                    "train/mDice": train_mdice,
                    "test/loss": test_loss,
                    "test/mIoU": test_miou,
                    "test/mDice": test_mdice,
                }
            )

    torch.save(model.state_dict(), output_dir / "unet_segmentation_23cls.pt")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_plots(history, str(output_dir))

    if run_wandb:
        wandb.save(str(output_dir / "training_curves.png"))
        wandb.save(str(output_dir / "history.json"))
        wandb.finish()

    print(f"Saved model to: {output_dir / 'unet_segmentation_23cls.pt'}")
    print(f"Saved metrics to: {output_dir / 'history.json'}")
    print(f"Saved curves to: {output_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
