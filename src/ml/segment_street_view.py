import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from .stego.src.train_segmentation import (
    LitUnsupervisedSegmenter,
    get_class_labels,
)
from .stego.src.utils import get_transform, unnorm, UnsupervisedMetrics
from .stego.src.crf import dense_crf


class ImageSegmenter:
    def __init__(self, stego_checkpoint_path):
        """
        Initialize the image segmenter
        Args:
            stego_checkpoint_path (str): Path to STEGO model checkpoint
        """
        # Load STEGO model
        self.model = LitUnsupervisedSegmenter.load_from_checkpoint(
            stego_checkpoint_path
        )

        # Select device (CUDA, MPS for Apple Silicon, or CPU fallback)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # for Apple Silicon
        else:
            self.device = torch.device("cpu")

        self.model.eval().to(self.device)

        # Set up image transform
        self.transform = get_transform(448, False, "center")

        # Get class labels
        self.class_labels = get_class_labels(self.model.cfg.dataset_name)

        # Initialize metrics
        self.n_classes = len(self.class_labels)
        self.cluster_metrics = UnsupervisedMetrics(
            prefix="cluster/",
            n_classes=self.n_classes,
            extra_clusters=self.model.cfg.extra_clusters,
            compute_hungarian=True,
        )

    def segment_image(self, image_path, output_dir):
        """
        Segment a single image and compute metrics
        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save results
        Returns:
            dict: Dictionary containing segmentation metrics and class distributions
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load and transform image
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Get STEGO predictions
        with torch.no_grad():
            code1 = self.model(img_tensor)
            code2 = self.model(img_tensor.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(
                code,
                img_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Get both linear and cluster predictions
            linear_probs = torch.log_softmax(
                self.model.linear_probe(code), dim=1
            ).cpu()
            cluster_probs = self.model.cluster_probe(
                code, 2, log_probs=True
            ).cpu()

            # Apply CRF refinement
            single_img = img_tensor[0].cpu()
            linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
            cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)

        # Compute metrics and class distributions
        metrics = self._compute_metrics(linear_pred)

        # Save results
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        self._save_results(
            img_tensor,
            linear_pred,
            cluster_pred,
            output_name,
            output_dir,
            metrics,
        )

        return metrics

    def segment_directory(self, input_dir, output_dir):
        """
        Segment all images in a directory
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save results
        Returns:
            dict: Dictionary containing aggregated metrics across all images
        """
        os.makedirs(output_dir, exist_ok=True)

        all_metrics = []

        # Process each image in directory
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_dir, filename)
                metrics = self.segment_image(image_path, output_dir)
                all_metrics.append(metrics)

        # Aggregate metrics across all images
        aggregated_metrics = self._aggregate_metrics(all_metrics)

        # Save aggregated metrics
        self._save_metrics_report(aggregated_metrics, output_dir)

        return aggregated_metrics

    def _compute_metrics(self, linear_pred):
        """
        Compute metrics for segmentation results
        Args:
            linear_pred: Linear predictions (numpy array or torch tensor)
        Returns:
            dict: Dictionary containing metrics
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(linear_pred, np.ndarray):
            linear_pred = torch.from_numpy(linear_pred)

        # Get pixel distribution for each class
        class_pixels = {}
        total_pixels = linear_pred.numel()

        # For linear predictions, we don't need Hungarian matching
        # since they already map directly to class indices
        
        # Compute pixel distribution for each class
        for class_idx in range(self.n_classes):
            pixels = (linear_pred == class_idx).sum().item()
            percentage = (pixels / total_pixels) * 100
            class_pixels[self.class_labels[class_idx]] = {
                "pixel_count": pixels,
                "percentage": percentage,
            }

        return {
            "class_distribution": class_pixels,
            "total_pixels": total_pixels,
        }

    def _aggregate_metrics(self, metrics_list):
        """
        Aggregate metrics across multiple images
        """
        total_pixels = sum(m["total_pixels"] for m in metrics_list)
        aggregated_distribution = {}

        # Initialize aggregated distribution
        for class_name in self.class_labels:
            aggregated_distribution[class_name] = {
                "pixel_count": 0,
                "percentage": 0,
            }

        # Sum up pixels for each class
        for metrics in metrics_list:
            for class_name, stats in metrics["class_distribution"].items():
                aggregated_distribution[class_name]["pixel_count"] += stats[
                    "pixel_count"
                ]

        # Calculate percentages
        for class_name in aggregated_distribution:
            pixel_count = aggregated_distribution[class_name]["pixel_count"]
            percentage = (pixel_count / total_pixels) * 100
            aggregated_distribution[class_name]["percentage"] = percentage

        return {
            "class_distribution": aggregated_distribution,
            "total_pixels": total_pixels,
            "n_images": len(metrics_list),
        }

    def _save_metrics_report(self, metrics, output_dir):
        """
        Save metrics report to a text file
        """
        report_path = os.path.join(output_dir, "segmentation_metrics.txt")

        with open(report_path, "w") as f:
            f.write("Segmentation Metrics Report\n")
            f.write("==========================\n\n")
            f.write(f"Total Images Processed: {metrics['n_images']}\n")
            f.write(f"Total Pixels Processed: {metrics['total_pixels']}\n\n")

            f.write("Class Distribution:\n")
            f.write("-----------------\n")

            # Sort classes by percentage for better readability
            sorted_classes = sorted(
                metrics["class_distribution"].items(),
                key=lambda x: x[1]["percentage"],
                reverse=True,
            )

            for class_name, stats in sorted_classes:
                f.write(f"{class_name}:\n")
                f.write(f"  Pixel Count: {stats['pixel_count']}\n")
                f.write(f"  Percentage: {stats['percentage']:.2f}%\n\n")

    def _save_results(
        self, img_tensor, linear_pred, cluster_pred, name, output_dir, metrics
    ):
        """
        Save segmentation results and metrics visualization with legends
        """
        fig = plt.figure(figsize=(24, 8))

        # Original image
        ax1 = plt.subplot(141)
        ax1.imshow(unnorm(img_tensor)[0].permute(1, 2, 0).cpu())
        ax1.set_title("Original Image", fontsize=12, fontweight='bold')
        ax1.axis("off")

        # Cluster predictions with legend
        ax2 = plt.subplot(142)
        cluster_img = ax2.imshow(self.model.label_cmap[cluster_pred])
        ax2.set_title("Cluster Predictions", fontsize=12, fontweight='bold')
        ax2.axis("off")
        
        # Add legend for cluster predictions
        self._add_segmentation_legend(ax2, cluster_pred, "Cluster")

        # Linear probe predictions with legend
        ax3 = plt.subplot(143)
        linear_img = ax3.imshow(self.model.label_cmap[linear_pred])
        ax3.set_title("Linear Probe Predictions", fontsize=12, fontweight='bold')
        ax3.axis("off")
        
        # Add legend for linear predictions
        self._add_segmentation_legend(ax3, linear_pred, "Linear")

        # Class distribution chart
        ax4 = plt.subplot(144)
        self._add_class_distribution_chart(ax4, metrics)

        # Save figure
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"segmentation_{name}.png"),
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def _add_segmentation_legend(self, ax, predictions, pred_type):
        """
        Add a legend showing class labels with their corresponding colors
        """
        # Get unique classes present in the predictions
        if hasattr(predictions, 'cpu'):
            # PyTorch tensor
            unique_classes = np.unique(predictions.cpu().numpy())
        else:
            # Already numpy array
            unique_classes = np.unique(predictions)
        
        # Create legend elements
        legend_elements = []
        for class_idx in unique_classes:
            if class_idx < len(self.class_labels):
                class_name = self.class_labels[class_idx]
                color = self.model.label_cmap[class_idx] / 255.0  # Normalize to [0,1]
                
                # Create a patch for the legend
                from matplotlib.patches import Patch
                legend_elements.append(
                    Patch(facecolor=color, label=f"{class_name}")
                )
        
        # Add legend to the right of the plot
        if legend_elements:
            ax.legend(
                handles=legend_elements,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=8,
                title=f"{pred_type} Classes",
                title_fontsize=9,
                frameon=True,
                fancybox=True,
                shadow=True
            )

    def _add_class_distribution_chart(self, ax, metrics):
        """
        Add a bar chart showing class distribution percentages
        """
        # Get sorted classes by percentage
        sorted_classes = sorted(
            metrics["class_distribution"].items(),
            key=lambda x: x[1]["percentage"],
            reverse=True,
        )
        
        # Take top 10 classes to avoid overcrowding
        top_classes = sorted_classes[:10]
        
        if top_classes:
            class_names = [item[0] for item in top_classes]
            percentages = [item[1]["percentage"] for item in top_classes]
            
            # Get colors for each class
            colors = []
            for class_name in class_names:
                class_idx = self.class_labels.index(class_name) if class_name in self.class_labels else 0
                color = self.model.label_cmap[class_idx] / 255.0
                colors.append(color)
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(class_names)), percentages, color=colors)
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=8)
            ax.set_xlabel("Percentage (%)", fontsize=9)
            ax.set_title("Class Distribution", fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add percentage labels on bars
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                ax.text(pct + 0.5, i, f"{pct:.1f}%", 
                       va='center', fontsize=7, fontweight='bold')
            
            # Invert y-axis to show highest percentage at top
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "No class data available", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Class Distribution", fontsize=12, fontweight='bold')

