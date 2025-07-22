import os
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from train_segmentation import LitUnsupervisedSegmenter, get_class_labels
from utils import get_transform, unnorm, UnsupervisedMetrics
from crf import dense_crf

class ImageSegmenter:
    def __init__(self, stego_checkpoint_path):
        """
        Initialize the image segmenter
        Args:
            stego_checkpoint_path (str): Path to STEGO model checkpoint
        """
        # Load STEGO model
        self.model = LitUnsupervisedSegmenter.load_from_checkpoint(stego_checkpoint_path)
        self.model.eval().cuda()
        
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
            compute_hungarian=True
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
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).cuda()
        
        # Get STEGO predictions
        with torch.no_grad():
            code1 = self.model(img_tensor)
            code2 = self.model(img_tensor.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2
            code = F.interpolate(code, img_tensor.shape[-2:], mode='bilinear', align_corners=False)
            
            # Get both linear and cluster predictions
            linear_probs = torch.log_softmax(self.model.linear_probe(code), dim=1).cpu()
            cluster_probs = self.model.cluster_probe(code, 2, log_probs=True).cpu()
            
            # Apply CRF refinement
            single_img = img_tensor[0].cpu()
            linear_pred = dense_crf(single_img, linear_probs[0]).argmax(0)
            cluster_pred = dense_crf(single_img, cluster_probs[0]).argmax(0)
        
        # Compute metrics and class distributions
        metrics = self._compute_metrics(cluster_pred)
        
        # Save results
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        self._save_results(img_tensor, linear_pred, cluster_pred, output_name, output_dir, metrics)
        
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
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                metrics = self.segment_image(image_path, output_dir)
                all_metrics.append(metrics)
        
        # Aggregate metrics across all images
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        # Save aggregated metrics
        self._save_metrics_report(aggregated_metrics, output_dir)
        
        return aggregated_metrics
    
    def _compute_metrics(self, cluster_pred):
        """
        Compute metrics for segmentation results
        Args:
            cluster_pred: Cluster predictions (numpy array or torch tensor)
        Returns:
            dict: Dictionary containing metrics
        """
        # Convert numpy array to torch tensor if needed
        if isinstance(cluster_pred, np.ndarray):
            cluster_pred = torch.from_numpy(cluster_pred)
            
        # Get pixel distribution for each class
        class_pixels = {}
        total_pixels = cluster_pred.numel()
        
        # Update metrics with current predictions
        # Create a dummy target tensor of zeros since we don't have ground truth
        dummy_target = torch.zeros_like(cluster_pred)
        self.cluster_metrics.update(cluster_pred, dummy_target)
        
        # Compute metrics to initialize assignments
        self.cluster_metrics.compute()
        
        # Map cluster predictions to class labels using Hungarian matching
        mapped_predictions = self.cluster_metrics.map_clusters(cluster_pred)
        
        # Compute pixel distribution for each class
        for class_idx in range(self.n_classes):
            pixels = (mapped_predictions == class_idx).sum().item()
            percentage = (pixels / total_pixels) * 100
            class_pixels[self.class_labels[class_idx]] = {
                'pixel_count': pixels,
                'percentage': percentage
            }
        
        return {
            'class_distribution': class_pixels,
            'total_pixels': total_pixels
        }
    
    def _aggregate_metrics(self, metrics_list):
        """
        Aggregate metrics across multiple images
        """
        total_pixels = sum(m['total_pixels'] for m in metrics_list)
        aggregated_distribution = {}
        
        # Initialize aggregated distribution
        for class_name in self.class_labels:
            aggregated_distribution[class_name] = {
                'pixel_count': 0,
                'percentage': 0
            }
        
        # Sum up pixels for each class
        for metrics in metrics_list:
            for class_name, stats in metrics['class_distribution'].items():
                aggregated_distribution[class_name]['pixel_count'] += stats['pixel_count']
        
        # Calculate percentages
        for class_name in aggregated_distribution:
            pixel_count = aggregated_distribution[class_name]['pixel_count']
            percentage = (pixel_count / total_pixels) * 100
            aggregated_distribution[class_name]['percentage'] = percentage
        
        return {
            'class_distribution': aggregated_distribution,
            'total_pixels': total_pixels,
            'n_images': len(metrics_list)
        }
    
    def _save_metrics_report(self, metrics, output_dir):
        """
        Save metrics report to a text file
        """
        report_path = os.path.join(output_dir, 'segmentation_metrics.txt')
        
        with open(report_path, 'w') as f:
            f.write("Segmentation Metrics Report\n")
            f.write("==========================\n\n")
            f.write(f"Total Images Processed: {metrics['n_images']}\n")
            f.write(f"Total Pixels Processed: {metrics['total_pixels']}\n\n")
            
            f.write("Class Distribution:\n")
            f.write("-----------------\n")
            
            # Sort classes by percentage for better readability
            sorted_classes = sorted(
                metrics['class_distribution'].items(),
                key=lambda x: x[1]['percentage'],
                reverse=True
            )
            
            for class_name, stats in sorted_classes:
                f.write(f"{class_name}:\n")
                f.write(f"  Pixel Count: {stats['pixel_count']}\n")
                f.write(f"  Percentage: {stats['percentage']:.2f}%\n\n")
    
    def _save_results(self, img_tensor, linear_pred, cluster_pred, name, output_dir, metrics):
        """
        Save segmentation results and metrics visualization
        """
        fig = plt.figure(figsize=(20, 5))
        
        # Original image
        ax1 = plt.subplot(131)
        ax1.imshow(unnorm(img_tensor)[0].permute(1, 2, 0).cpu())
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Cluster predictions
        ax2 = plt.subplot(132)
        ax2.imshow(self.model.label_cmap[cluster_pred])
        ax2.set_title("Cluster Predictions")
        ax2.axis('off')
        
        # Linear probe predictions
        ax3 = plt.subplot(133)
        ax3.imshow(self.model.label_cmap[linear_pred])
        ax3.set_title("Linear Probe Predictions")
        ax3.axis('off')
        
        # Add metrics text
        plt.figtext(0.02, 0.02, "Class Distribution:", fontsize=8)
        text_y = 0.15
        sorted_classes = sorted(
            metrics['class_distribution'].items(),
            key=lambda x: x[1]['percentage'],
            reverse=True
        )
        for class_name, stats in sorted_classes[:5]:  # Show top 5 classes
            plt.figtext(0.02, text_y, 
                       f"{class_name}: {stats['percentage']:.1f}%",
                       fontsize=8)
            text_y += 0.03
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'segmentation_{name}.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    # Example usage
    STEGO_CHECKPOINT = "saved_models/cityscapes_vit_base_1.ckpt"  # Path to STEGO checkpoint
    #STEGO_CHECKPOINT = "saved_models/cocostuff27_vit_base_5.ckpt"  # Path to STEGO checkpoint
    # Initialize segmenter
    segmenter = ImageSegmenter(STEGO_CHECKPOINT)
    
    # Example: Segment a single image
    image_path = "test_images/buenos_aires_view_2015.jpg"  # Replace with your image path
    metrics = segmenter.segment_image(image_path, "segmentation_results")
    print("\nSingle Image Metrics:")
    print("Class Distribution:")
    for class_name, stats in metrics['class_distribution'].items():
        print(f"{class_name}: {stats['percentage']:.2f}%")
    
    # Example: Segment all images in a directory
    # input_dir = "test_images"
    # aggregated_metrics = segmenter.segment_directory(input_dir, "segmentation_results")
    # print("\nAggregated Metrics:")
    # print(f"Total Images: {aggregated_metrics['n_images']}")
    # print("Average Class Distribution:")
    # for class_name, stats in aggregated_metrics['class_distribution'].items():
    #     print(f"{class_name}: {stats['percentage']:.2f}%") 