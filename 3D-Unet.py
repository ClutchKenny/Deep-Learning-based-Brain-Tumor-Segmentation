
from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    ToTensord,
    DivisiblePadd,
)
from monai.data import Dataset

from sklearn.model_selection import KFold
from medpy.metric.binary import hd95


def compute_dice(pred, target):
    """Compute Dice score for a single binary mask."""
    smooth = 1e-5
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def compute_brats_metrics(pred, target):
    """
    Compute Dice and HD95 for ET, TC, WT regions.

    pred: predicted segmentation (H, W, D) with values 0-3
    target: ground truth (H, W, D) with values 0, 1, 2, 4
    """
    results = {}

    # ET: Enhancing Tumor (label 4 in target, class 3 in pred)
    et_pred = (pred == 3)
    et_target = (target == 4)

    # TC: Tumor Core (labels 1 + 4 in target, classes 1 + 3 in pred)
    tc_pred = (pred == 1) | (pred == 3)
    tc_target = (target == 1) | (target == 4)

    # WT: Whole Tumor (labels 1 + 2 + 4 in target, classes 1 + 2 + 3 in pred)
    wt_pred = (pred == 1) | (pred == 2) | (pred == 3)
    wt_target = (target == 1) | (target == 2) | (target == 4)

    for name, (p, t) in [("ET", (et_pred, et_target)),
                          ("TC", (tc_pred, tc_target)),
                          ("WT", (wt_pred, wt_target))]:
        p_np = p.cpu().numpy().astype(bool)
        t_np = t.cpu().numpy().astype(bool)

        # Dice score
        dice = compute_dice(p_np, t_np)

        # Hausdorff distance 95 (From medpy)
        if p_np.sum() > 0 and t_np.sum() > 0:
            hausdorff = hd95(p_np, t_np)
        else:
            hausdorff = np.nan

        results[name] = {"dice": dice, "hd95": hausdorff}

    return results


def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Trains the model for one epoch.
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: DataLoader for training samples.
    optimizer: Optimizer to use for model parameter updates.
    criterion: Loss function (DiceLoss).
    epoch: Current epoch number.
    """
    model.train()

    losses = []

    for batch_idx, batch_sample in enumerate(train_loader):
        data = batch_sample["image"].to(device)
        target = batch_sample["label"].to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        losses.append(loss.item())
        optimizer.step()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}')

    train_loss = float(np.mean(losses))
    print(f'Train set: Average loss: {train_loss:.4f}')

    return train_loss


def evaluate(model, device, loader, criterion, split_name='Val'):
    """
    Evaluates the model on a given data loader.
    model: The model to evaluate. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    loader: DataLoader for the split to evaluate.
    criterion: Loss function.
    split_name: Label for print output ('Val' or 'Test').
    """
    model.eval()

    losses = []
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(loader):
            data = batch_sample["image"].to(device)
            target = batch_sample["label"].to(device)

            output = model(data)
            loss = criterion(output, target)
            losses.append(loss.item())

            # Get predictions and compute metrics
            preds = torch.argmax(output, dim=1).squeeze()
            target_squeezed = target.squeeze()

            metrics = compute_brats_metrics(preds, target_squeezed)
            all_metrics.append(metrics)

    eval_loss = float(np.mean(losses))

    # Average metrics across all samples
    avg_metrics = {}
    for region in ["ET", "TC", "WT"]:
        avg_metrics[region] = {
            "dice": np.nanmean([m[region]["dice"] for m in all_metrics]),
            "hd95": np.nanmean([m[region]["hd95"] for m in all_metrics]),
        }

    print(f'{split_name} set: Average loss: {eval_loss:.4f}')
    print(f'  ET: Dice={avg_metrics["ET"]["dice"]:.4f}, HD95={avg_metrics["ET"]["hd95"]:.2f}')
    print(f'  TC: Dice={avg_metrics["TC"]["dice"]:.4f}, HD95={avg_metrics["TC"]["hd95"]:.2f}')
    print(f'  WT: Dice={avg_metrics["WT"]["dice"]:.4f}, HD95={avg_metrics["WT"]["hd95"]:.2f}\n')

    return eval_loss, avg_metrics


def save_results(all_fold_results, output_path='brats_results.csv'):
    """
    Saves results to CSV file.
    all_fold_results: List of metric dictionaries for each fold.
    output_path: Path to save CSV file.
    """
    import pandas as pd

    rows = []
    for fold_idx, fold_result in enumerate(all_fold_results):
        rows.append({
            "Fold": fold_idx + 1,
            "ET_Dice": fold_result["ET"]["dice"],
            "ET_HD95": fold_result["ET"]["hd95"],
            "TC_Dice": fold_result["TC"]["dice"],
            "TC_HD95": fold_result["TC"]["hd95"],
            "WT_Dice": fold_result["WT"]["dice"],
            "WT_HD95": fold_result["WT"]["hd95"],
        })

    # Add average row
    rows.append({
        "Fold": "AVG",
        "ET_Dice": np.nanmean([r["ET"]["dice"] for r in all_fold_results]),
        "ET_HD95": np.nanmean([r["ET"]["hd95"] for r in all_fold_results]),
        "TC_Dice": np.nanmean([r["TC"]["dice"] for r in all_fold_results]),
        "TC_HD95": np.nanmean([r["TC"]["hd95"] for r in all_fold_results]),
        "WT_Dice": np.nanmean([r["WT"]["dice"] for r in all_fold_results]),
        "WT_HD95": np.nanmean([r["WT"]["hd95"] for r in all_fold_results]),
    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")

    return df


def plot_training_curves(train_losses, val_losses, val_metrics, fold, output_dir='plots'):
    """
    Plots training curves for a single fold.
    train_losses: List of training losses per epoch.
    val_losses: List of validation losses per epoch.
    val_metrics: List of validation metrics per epoch.
    fold: Fold number.
    output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Loss plot
    axes[0].plot(epochs, train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, label='Val Loss', linewidth=2)
    axes[0].set_title(f'Fold {fold} - Loss per Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Dice Loss')
    axes[0].grid(True)
    axes[0].legend(loc='best')

    # Dice score plot
    et_dice = [m["ET"]["dice"] for m in val_metrics]
    tc_dice = [m["TC"]["dice"] for m in val_metrics]
    wt_dice = [m["WT"]["dice"] for m in val_metrics]

    axes[1].plot(epochs, et_dice, label='ET Dice', linewidth=2)
    axes[1].plot(epochs, tc_dice, label='TC Dice', linewidth=2)
    axes[1].plot(epochs, wt_dice, label='WT Dice', linewidth=2)
    axes[1].set_title(f'Fold {fold} - Dice Score per Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].grid(True)
    axes[1].legend(loc='best')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fold{fold}_curves.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Training curves saved to '{out_path}'")


def visualize_prediction(model, device, val_loader, fold, slice_idx=77, output_dir='visualizations'):
    """
    Saves visualization of model prediction vs ground truth.
    model: Trained model.
    device: 'cuda' or 'cpu'.
    val_loader: DataLoader for validation samples.
    fold: Fold number.
    slice_idx: Which slice to visualize.
    output_dir: Directory to save visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    sample = next(iter(val_loader))
    image = sample["image"].to(device)
    label = sample["label"].squeeze().cpu().numpy()

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Adjust slice index if needed
    max_slice = min(label.shape[-1], pred.shape[-1]) - 1
    slice_idx = min(slice_idx, max_slice)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # MRI slice (first modality - FLAIR)
    axes[0].imshow(image[0, 0, :, :, slice_idx].cpu().numpy(), cmap='gray')
    axes[0].set_title("MRI (FLAIR)")
    axes[0].axis('off')

    # Ground truth
    axes[1].imshow(label[:, :, slice_idx], cmap='viridis', vmin=0, vmax=4)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    # Prediction
    axes[2].imshow(pred[:, :, slice_idx], cmap='viridis', vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'fold{fold}_prediction.png')
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    print(f"Visualization saved to '{out_path}'")


def run_main(FLAGS):
    """Main training and evaluation function."""

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Torch device selected: {device}")

    # -----------------------------------------------------------------------
    # Load Data Paths
    # -----------------------------------------------------------------------
    data_dir = FLAGS.data_dir
    images_dir = os.path.join(data_dir, "imagesTr")
    labels_dir = os.path.join(data_dir, "labelsTr")

    # Filter out hidden files (Mac metadata)
    image_files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith('.nii.gz') and not f.startswith('.')
    ])

    label_files = sorted([
        os.path.join(labels_dir, f)
        for f in os.listdir(labels_dir)
        if f.endswith('.nii.gz') and not f.startswith('.')
    ])

    data_dicts = [
        {"image": img, "label": lbl}
        for img, lbl in zip(image_files, label_files)
    ]

    print(f"Found {len(data_dicts)} samples")

    # -----------------------------------------------------------------------
    # Define Transforms (From MONAI)
    # -----------------------------------------------------------------------
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(keys=["image", "label"], roi_size=FLAGS.roi_size, random_size=False),
        DivisiblePadd(keys=["image", "label"], k=16),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=16),
        ToTensord(keys=["image", "label"]),
    ])

    # -----------------------------------------------------------------------
    # Build 3D U-Net Model
    # -----------------------------------------------------------------------
    print(f"\nBuilding 3D U-Net:")
    print(f"  Input channels: 4 (FLAIR, T1, T1ce, T2)")
    print(f"  Output channels: 4 (Background, NCR/NET, ED, ET)")
    print(f"  ROI size: {FLAGS.roi_size}")

    # -----------------------------------------------------------------------
    # Loss and Optimizer Setup
    # -----------------------------------------------------------------------
    criterion = DiceLoss(to_onehot_y=True, softmax=True)

    # -----------------------------------------------------------------------
    # 5-Fold Cross-Validation
    # -----------------------------------------------------------------------
    all_fold_results = []
    kfold = KFold(n_splits=FLAGS.num_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data_dicts)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{FLAGS.num_folds}")
        print(f"{'='*60}")

        # Split data
        train_data = [data_dicts[i] for i in train_idx]
        val_data = [data_dicts[i] for i in val_idx]
        print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

        # Create datasets
        train_ds = Dataset(data=train_data, transform=train_transforms)
        val_ds = Dataset(data=val_data, transform=val_transforms)

        # Create dataloaders
        train_loader = DataLoader(
            train_ds,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=FLAGS.num_workers
        )

        # Create fresh model for each fold (From MONAI)
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

        # Track metrics for plotting
        train_losses = []
        val_losses = []
        val_metrics_history = []
        best_dice = 0.0

        # Training loop
        for epoch in range(1, FLAGS.num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{FLAGS.num_epochs} ---")

            train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
            val_loss, val_metrics = evaluate(model, device, val_loader, criterion, 'Val')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_metrics_history.append(val_metrics)

            # Track best model
            avg_dice = (val_metrics["ET"]["dice"] + val_metrics["TC"]["dice"] + val_metrics["WT"]["dice"]) / 3
            if avg_dice > best_dice:
                best_dice = avg_dice
                if FLAGS.save_model:
                    torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')

        # Store final metrics for this fold
        all_fold_results.append(val_metrics)

        # Plot training curves
        if FLAGS.save_plots:
            plot_training_curves(train_losses, val_losses, val_metrics_history, fold + 1)

        # Save visualization
        if FLAGS.save_visualizations:
            visualize_prediction(model, device, val_loader, fold + 1)

        print(f"\nFold {fold + 1} Best Average Dice: {best_dice:.4f}")

    # -----------------------------------------------------------------------
    # Final Results
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("FINAL RESULTS - ALL FOLDS")
    print(f"{'='*60}")

    # Print results table
    print(f"\n{'Fold':<8} {'ET Dice':<10} {'ET HD95':<10} {'TC Dice':<10} {'TC HD95':<10} {'WT Dice':<10} {'WT HD95':<10}")
    print("-" * 68)

    for fold_idx, fold_result in enumerate(all_fold_results):
        print(f"{fold_idx+1:<8} "
              f"{fold_result['ET']['dice']:<10.4f} "
              f"{fold_result['ET']['hd95']:<10.2f} "
              f"{fold_result['TC']['dice']:<10.4f} "
              f"{fold_result['TC']['hd95']:<10.2f} "
              f"{fold_result['WT']['dice']:<10.4f} "
              f"{fold_result['WT']['hd95']:<10.2f}")

    print("-" * 68)

    # Print averages
    avg_et_dice = np.nanmean([r["ET"]["dice"] for r in all_fold_results])
    avg_et_hd = np.nanmean([r["ET"]["hd95"] for r in all_fold_results])
    avg_tc_dice = np.nanmean([r["TC"]["dice"] for r in all_fold_results])
    avg_tc_hd = np.nanmean([r["TC"]["hd95"] for r in all_fold_results])
    avg_wt_dice = np.nanmean([r["WT"]["dice"] for r in all_fold_results])
    avg_wt_hd = np.nanmean([r["WT"]["hd95"] for r in all_fold_results])

    print(f"{'AVG':<8} "
          f"{avg_et_dice:<10.4f} "
          f"{avg_et_hd:<10.2f} "
          f"{avg_tc_dice:<10.4f} "
          f"{avg_tc_hd:<10.2f} "
          f"{avg_wt_dice:<10.4f} "
          f"{avg_wt_hd:<10.2f}")

    # Save results to CSV
    if FLAGS.save_results:
        save_results(all_fold_results, FLAGS.output_csv)

    print("\nTraining and evaluation finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('3D U-Net Brain Tumor Segmentation (BraTS)')

    parser.add_argument('--data_dir',
                        type=str,
                        default='/content/data/Task01_BrainTumour',
                        help='Path to BraTS dataset directory.')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=25,
                        help='Number of epochs to train.')
    parser.add_argument('--num_folds',
                        type=int,
                        default=5,
                        help='Number of folds for cross-validation.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size for training.')
    parser.add_argument('--roi_size',
                        type=int,
                        nargs=3,
                        default=[128, 128, 128],
                        help='ROI size for random cropping (D H W).')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of data loading workers.')
    parser.add_argument('--save_model',
                        action='store_true',
                        help='Save best model for each fold.')
    parser.add_argument('--save_plots',
                        action='store_true',
                        help='Save training curves.')
    parser.add_argument('--save_visualizations',
                        action='store_true',
                        help='Save prediction visualizations.')
    parser.add_argument('--save_results',
                        action='store_true',
                        help='Save results to CSV.')
    parser.add_argument('--output_csv',
                        type=str,
                        default='brats_results.csv',
                        help='Output CSV file path.')

    FLAGS, unparsed = parser.parse_known_args()

    # Convert roi_size to tuple
    FLAGS.roi_size = tuple(FLAGS.roi_size)

    run_main(FLAGS)