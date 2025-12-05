#!/usr/bin/env python3
# cifar_pipeline.py - Complete pipeline for CIFAR-10/100 data preparation, augmentation, training and evaluation

import argparse
import logging
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # Imports related to layout
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

# Import our custom modules
from scripts.data_download import download_and_extract_cifar10_data, download_and_extract_cifar100_data
from scripts.data_augmentation import augment_dataset
from scripts.model_architectures import create_model
from scripts.train_utils import (
    save_metrics,
    train_epoch,
    validate_epoch,
    save_checkpoint,
    load_checkpoint,
    define_loss_and_optimizer,
    load_data,
    load_transforms,
)
from scripts.evaluation_metrics import (
    evaluate_model,
    top_k_accuracy,
    visualize_predictions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("cifar_pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CIFAR-10/100 Training Pipeline")

    # Dataset selection
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar100",
                        help="Dataset to use (cifar10 or cifar100)")

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Base directory for data storage")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")

    # Data augmentation
    parser.add_argument("--aug_count", type=int, default=0,
                        help="Number of augmentations per image")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=60,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay (L2 penalty)")

    # Checkpointing
    parser.add_argument("--save_freq", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=100,
                        help="Early stopping patience")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    # Random seeds
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    return parser.parse_args()

def collect_data(args):
    """Collect data"""
    logger.info(f"Collecting {args.dataset} dataset...")

    # Create the directory for our raw data if it doesn't already exist
    print("Preparing data directory...")
    os.makedirs(os.path.join(args.data_dir, "raw"), exist_ok=True)
    print("Setup complete.")

    if args.dataset == "cifar10":
        train_dataset, test_dataset = download_and_extract_cifar10_data(
            root_dir=os.path.join(args.data_dir, "raw"),
        )
    else:
        train_dataset, test_dataset = download_and_extract_cifar100_data(
            root_dir=os.path.join(args.data_dir, "raw"),
        )

def augment_data(args):
    """Prepare and augment data"""
    logger.info(f"Augmenting {args.dataset} dataset...")

    raw_data_dir = os.path.join(args.data_dir, "raw", "train")
    augmented_data_dir = os.path.join(args.data_dir, "augmented", "train")
    augmentations_per_image = args.aug_count

    # --- Path Validation ---
    if not os.path.exists(raw_data_dir):
        print(f"❌ Error: Raw data directory '{raw_data_dir}' not found.")
        print("Please ensure you have run 'collect_data' first.")
    else:
        print(f"✅ Found raw data at: {raw_data_dir}")
        print(f"   Augmented data will be saved to: {augmented_data_dir}")
        print(f"   Number of augmentations per image: {augmentations_per_image}")

    if os.path.exists(raw_data_dir):
        print("🚀 Starting data augmentation...")
        augment_dataset(
            input_dir=raw_data_dir,
            output_dir=augmented_data_dir,
            augmentations_per_image=augmentations_per_image
        )
        print("\n🎉 Data augmentation completed successfully!")
    else:
        print("Skipping augmentation process due to missing raw data directory.")

    return augmented_data_dir

def build_model(args):
    """Build the model"""
    num_classes = 10 if args.dataset == "cifar10" else 100
    logger.info(f"Creating model with {num_classes} classes, {args.device} device...")
    model = create_model(num_classes=num_classes, device=args.device)
    return model

def train(args, model: nn.Module):
    # Define loss and optimizer
    criterion, optimizer, scheduler = define_loss_and_optimizer(model, args.lr, args.weight_decay)

    # Initialize tracking variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Lists to store training history for later plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Create directories for saving models and results if they don't exist
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    print(f"Training configured for {args.num_epochs} epochs with early stopping patience of {args.early_stopping_patience}.")

    # Load data
    train_loader, val_loader = load_data(os.path.join(args.data_dir, "augmented", "train"), args.batch_size)

    print("Starting training...")
    for epoch in range(args.num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)

        # Validate the model
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, args.device)

        # Update learning rate (SequentialLR scheduler)
        scheduler.step()

        # Store metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Check for improvement and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                os.path.join(args.output_dir, "models", "best_model.pth"),
            )
            print("  ↳ Validation loss improved. Saving best model!")
        else:
            patience_counter += 1
            print(
                f"  ↳ No improvement. Early stopping counter: {patience_counter}/{args.early_stopping_patience}"
            )

        # Check for early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    print("\nTraining completed!")

    # Load the best model checkpoint
    best_model_path = os.path.join(args.output_dir, "models", "best_model.pth")
    checkpoint = load_checkpoint(best_model_path, model, optimizer, scheduler)
    model.load_state_dict(checkpoint["state_dict"])

    # Retrieve details from the checkpoint
    best_epoch = checkpoint["epoch"]
    best_val_loss_loaded = checkpoint["best_val_loss"]

    print(f"Loaded best model from epoch {best_epoch} with validation loss {best_val_loss_loaded:.4f}")

    # Save final model state_dict
    torch.save(model.state_dict(), os.path.join(args.output_dir, "models", "final_model.pth"))
    print(f"Final model state_dict saved to '{os.path.join(args.output_dir, 'models', 'final_model.pth')}'.")

    return model, best_val_loss

# Custom confusion matrix plotting function (fixes colorbar issue)
def plot_custom_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    # Normalization (for better visualization)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap and return im object (for colorbar usage)
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
    plt.yticks(tick_marks, class_names, fontsize=6)
    
    # Set labels
    plt.xlabel('Predicted Label', fontsize=8)
    plt.ylabel('True Label', fontsize=8)
    
    return im

# Custom ROC curve plotting function (returns legend handles and labels)
def plot_custom_roc_curves(ax, y_true, y_probs, class_names):
    # Binarize labels (One-vs-Rest)
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    n_classes = y_true_bin.shape[1]
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    handles = []  # Store curve legend handles
    labels = []   # Store curve legend labels
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Plot curve and collect handles and labels
        curve_handle, = ax.plot(fpr[i], tpr[i], lw=0.5)
        handles.append(curve_handle)
        labels.append(f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot reference line for random guess
    rand_handle, = ax.plot([0, 1], [0, 1], 'k--', lw=1)
    handles.append(rand_handle)
    labels.append('Random Guess (AUC = 0.50)')
    
    # Set axis
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=8)
    ax.set_ylabel('True Positive Rate', fontsize=8)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=12)
    
    return handles, labels

def evaluate(args, model: nn.Module):
    """Evaluate the model on test data and generate all metrics"""
    # Create results directory if not exists
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load test dataset
    test_data_dir = os.path.join(args.data_dir, "raw", "test")
    test_transforms = load_transforms()
    test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    class_names = test_dataset.classes  # Get class names for metrics

    # Set model to evaluation mode
    model.eval()

    # Define loss function
    criterion, _, _ = define_loss_and_optimizer(model, args.lr, args.weight_decay)

    # Get evaluation results (Core: Back up variables first to prevent modification)
    test_loss, test_accuracy, all_preds, all_labels, all_probs = evaluate_model(
        model, test_loader, criterion, args.device
    )
    
    # Force convert to numpy array and back up to prevent modification by subsequent functions
    all_labels_np = np.array(all_labels).flatten()
    all_preds_np = np.array(all_preds).flatten()
    all_probs_np = np.array(all_probs)
    
    # Length check (critical fix)
    print(f"Labels length: {len(all_labels_np)}, Predictions length: {len(all_preds_np)}")
    if len(all_labels_np) != len(all_preds_np):
        # Truncate to the shorter length (prevent errors)
        min_len = min(len(all_labels_np), len(all_preds_np))
        all_labels_np = all_labels_np[:min_len]
        all_preds_np = all_preds_np[:min_len]
        all_probs_np = all_probs_np[:min_len]
        print(f"⚠️  Length mismatch! Truncated to min length: {min_len}")

    # -------------------------- 1. Classification Report --------------------------
    metrics_str = classification_report(
        all_labels_np, all_preds_np, target_names=class_names, zero_division=0
    )
    full_report_path = os.path.join(results_dir, "full_evaluation.txt")
    with open(full_report_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(metrics_str + "\n\n")
    print("Classification Report saved.")

    # -------------------------- 2. Top-K Accuracy --------------------------
    top1_acc = top_k_accuracy(all_labels_np, all_probs_np, k=1)
    top5_acc = top_k_accuracy(all_labels_np, all_probs_np, k=5)
    topk_str = (f"=== Top-K Accuracy ===\n"
                f"Top-1 Accuracy (Exact Match): {top1_acc:.2f}%\n"
                f"Top-5 Accuracy (Correct in Top 5): {top5_acc:.2f}%\n\n")
    print(topk_str)
    with open(full_report_path, "a") as f:
        f.write(topk_str)

    # -------------------------- 3. Confusion Matrix (Fix colorbar + style optimization) --------------------------
    plt.figure(figsize=(25, 25))  # Adapt to 100 classes of CIFAR100
    im = plot_custom_confusion_matrix(all_labels_np, all_preds_np, class_names)
    plt.title("Confusion Matrix (Normalized)", fontsize=12)
    
    # Pass im object to create colorbar
    cbar = plt.colorbar(im, shrink=0.8, aspect=40)
    cbar.ax.tick_params(labelsize=8)  # Adjust colorbar font size
    
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")

    # -------------------------- 4. ROC Curves (Ultimate compactness: two columns of legends almost overlap) --------------------------
    fig = plt.figure(figsize=(20, 10))  # Extremely narrow canvas to force legend compression
    # Overall layout: larger proportion for curve area, narrower legend area (4:1) to force compactness
    gs_main = GridSpec(1, 2, width_ratios=[4, 1])
    
    # Left column: ROC curves (occupies most space)
    ax_roc = fig.add_subplot(gs_main[0, 0])
    all_handles, all_labels_roc = plot_custom_roc_curves(ax_roc, all_labels_np, all_probs_np, class_names)

    # Right column: split into 1 row and 2 columns with extreme overlap (wspace=-0.3, deep overlap)
    gs_legend = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0, 1], wspace=-0.3)
    
    # First sub-axis on the right (left half of legend) - deep right shift
    ax_legend_left = fig.add_subplot(gs_legend[0, 0])
    ax_legend_left.axis("off")
    ax_legend_left.legend(
        all_handles[:len(all_handles)//2],
        all_labels_roc[:len(all_labels_roc)//2],
        loc="center", fontsize="xx-small", ncol=1,
        bbox_to_anchor=(1.5, 0.5),  # Extreme right shift, close to the right legend
        frameon=False  # Remove legend frame to reduce visual spacing
    )
    
    # Second sub-axis on the right (right half of legend) - deep left shift
    ax_legend_right = fig.add_subplot(gs_legend[0, 1])
    ax_legend_right.axis("off")
    ax_legend_right.legend(
        all_handles[len(all_handles)//2:],
        all_labels_roc[len(all_labels_roc)//2:],
        loc="center", fontsize="xx-small", ncol=1,
        bbox_to_anchor=(-0.5, 0.5),  # Extreme left shift, close to the left legend
        frameon=False  # Remove legend frame to reduce visual spacing
    )

    # No margins at all, force to fill the canvas
    plt.tight_layout(pad=0, rect=[0, 0, 1, 1])

    roc_path = os.path.join(results_dir, "roc_curves.png")
    plt.savefig(roc_path, bbox_inches=False, dpi=150)  # Close bbox_inches to further compress
    plt.close()
    print(f"ROC Curves saved to {roc_path}")

    # -------------------------- 5. Visualize Predictions --------------------------
    visualize_predictions(
        model,
        test_loader,
        args.device,
        class_names,
        num_samples=10
    )
    print(f"Prediction visualizations saved to {results_dir}/correct_predictions.png and {results_dir}/incorrect_predictions.png")

    # -------------------------- 6. Error Analysis --------------------------
    try:
        cm = confusion_matrix(all_labels_np, all_preds_np)
        np.fill_diagonal(cm, 0)
        indices = np.dstack(np.unravel_index(np.argsort(cm.ravel()), cm.shape))[0]

        error_str = "=== Top 10 Most Common Misclassifications ===\n"
        count = 0
        for i, j in reversed(indices):
            if cm[i, j] > 0:
                error_str += f"'{class_names[i]}' misclassified as '{class_names[j]}': {cm[i, j]} times\n"
                count += 1
                if count >= 10:
                    break
        print(error_str)
        with open(full_report_path, "a") as f:
            f.write(error_str)
    except Exception as e:
        error_msg = f"⚠️  Error analysis failed: {str(e)}\n"
        print(error_msg)
        with open(full_report_path, "a") as f:
            f.write(error_msg)

    print(f"All evaluation results saved to {results_dir}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # Set random seeds
    set_random_seeds(args.seed)

    # Print configuration
    logger.info("Starting CIFAR pipeline with configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Collect data (uncomment to enable)
    collect_data(args)
    
    # Augment data (uncomment to enable)
    augment_data(args)
    
    # Build model
    model = build_model(args)
    
    # Train
    train(args, model)
    
    # Evaluate
    evaluate(args, model)

if __name__ == "__main__":
    main()
