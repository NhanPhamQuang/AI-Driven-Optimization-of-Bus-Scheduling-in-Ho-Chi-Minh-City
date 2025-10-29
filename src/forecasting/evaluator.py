"""
Evaluation metrics for demand forecasting
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

from .config import ForecastingConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates forecasting model performance"""

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        self.config.ensure_dirs()

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true, y_pred)

        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

        # R² Score
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        max_error = np.max(np.abs(y_true - y_pred))
        median_ae = np.median(np.abs(y_true - y_pred))

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Max_Error': max_error,
            'Median_AE': median_ae
        }

        return metrics

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate predictions and log results

        Args:
            y_true: Actual values
            y_pred: Predicted values
            dataset_name: Name of dataset being evaluated

        Returns:
            Dictionary of metrics
        """
        logger.info(f"\nEvaluating {dataset_name} Set:")
        logger.info("=" * 60)

        metrics = self.calculate_metrics(y_true, y_pred)

        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        # Check against target accuracy
        if metrics['MAPE'] <= self.config.TARGET_MAPE:
            logger.info(f"\n✓ Target accuracy achieved! MAPE {metrics['MAPE']:.2f}% <= {self.config.TARGET_MAPE}%")
        else:
            logger.warning(f"\n✗ Target accuracy not achieved. MAPE {metrics['MAPE']:.2f}% > {self.config.TARGET_MAPE}%")

        logger.info("=" * 60)

        return metrics

    def plot_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot training history

        Args:
            history: Training history dictionary
            save_path: Path to save plot
        """
        if not self.config.PLOT_TRAINING_HISTORY:
            return

        logger.info("Plotting training history")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')

        # Loss
        axes[0, 0].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # MAE
        axes[0, 1].plot(history['mae'], label='Train MAE', linewidth=2)
        axes[0, 1].plot(history['val_mae'], label='Val MAE', linewidth=2)
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # RMSE
        axes[1, 0].plot(history['rmse'], label='Train RMSE', linewidth=2)
        axes[1, 0].plot(history['val_rmse'], label='Val RMSE', linewidth=2)
        axes[1, 0].set_title('Root Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate (if available)
        axes[1, 1].plot(range(1, len(history['loss']) + 1), linewidth=2)
        axes[1, 1].set_title('Training Progress')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Epoch Number')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.config.SAVE_PLOTS:
            save_path = save_path or os.path.join(self.config.PLOTS_DIR, 'training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")

        plt.close()

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None,
        n_samples: int = 500
    ):
        """
        Plot predictions vs actual values

        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save plot
            n_samples: Number of samples to plot
        """
        if not self.config.PLOT_PREDICTIONS:
            return

        logger.info("Plotting predictions")

        # Limit samples for better visualization
        if len(y_true) > n_samples:
            indices = np.linspace(0, len(y_true) - 1, n_samples).astype(int)
            y_true = y_true[indices]
            y_pred = y_pred[indices]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Time series plot
        axes[0, 0].plot(y_true, label='Actual', linewidth=2, alpha=0.7)
        axes[0, 0].plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Time Series: Actual vs Predicted')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Passengers')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.5)
        axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_title('Scatter: Actual vs Predicted')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Residuals
        residuals = y_true - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title('Residual Plot')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)

        # Error distribution
        errors = np.abs(residuals)
        axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Absolute Error Distribution')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.config.SAVE_PLOTS:
            save_path = save_path or os.path.join(self.config.PLOTS_DIR, 'predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved predictions plot to {save_path}")

        plt.close()

    def plot_error_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot detailed error analysis

        Args:
            y_true: Actual values
            y_pred: Predicted values
            save_path: Path to save plot
        """
        logger.info("Plotting error analysis")

        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors / (y_true + 1e-10)) * 100

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Detailed Error Analysis', fontsize=16, fontweight='bold')

        # Absolute errors over time
        axes[0, 0].plot(abs_errors, linewidth=1, alpha=0.7)
        axes[0, 0].set_title('Absolute Errors Over Time')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].grid(True, alpha=0.3)

        # Percentage errors over time
        axes[0, 1].plot(pct_errors, linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Percentage Errors Over Time')
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('Percentage Error (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Error distribution
        axes[0, 2].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].set_xlabel('Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)

        # Absolute error distribution
        axes[1, 0].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Absolute Error Distribution')
        axes[1, 0].set_xlabel('Absolute Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        # Percentage error distribution
        axes[1, 1].hist(pct_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('Percentage Error Distribution')
        axes[1, 1].set_xlabel('Percentage Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        # QQ plot for normality
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot (Error Normality)')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.config.SAVE_PLOTS:
            save_path = save_path or os.path.join(self.config.PLOTS_DIR, 'error_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error analysis plot to {save_path}")

        plt.close()

    def generate_report(
        self,
        metrics: Dict[str, float],
        dataset_name: str = "Test"
    ) -> str:
        """
        Generate evaluation report

        Args:
            metrics: Metrics dictionary
            dataset_name: Name of dataset

        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"EVALUATION REPORT - {dataset_name} SET")
        report.append("=" * 80)
        report.append("")

        report.append("Performance Metrics:")
        report.append("-" * 80)
        report.append(f"  Mean Absolute Error (MAE):           {metrics['MAE']:.4f}")
        report.append(f"  Root Mean Squared Error (RMSE):      {metrics['RMSE']:.4f}")
        report.append(f"  Mean Absolute Percentage Error:      {metrics['MAPE']:.2f}%")
        report.append(f"  R² Score:                            {metrics['R2']:.4f}")
        report.append(f"  Maximum Error:                       {metrics['Max_Error']:.4f}")
        report.append(f"  Median Absolute Error:               {metrics['Median_AE']:.4f}")
        report.append("")

        # Accuracy assessment
        report.append("Accuracy Assessment:")
        report.append("-" * 80)
        accuracy = 100 - metrics['MAPE']
        report.append(f"  Model Accuracy:                      {accuracy:.2f}%")
        report.append(f"  Target Accuracy:                     {100 - self.config.TARGET_MAPE:.2f}%")

        if accuracy >= (100 - self.config.TARGET_MAPE):
            report.append(f"  Status:                              ✓ TARGET ACHIEVED")
        else:
            report.append(f"  Status:                              ✗ BELOW TARGET")

        report.append("")
        report.append("=" * 80)

        report_str = "\n".join(report)
        logger.info("\n" + report_str)

        return report_str

    def save_report(
        self,
        report: str,
        filename: str = "evaluation_report.txt"
    ):
        """Save evaluation report to file"""
        filepath = os.path.join(self.config.PLOTS_DIR, filename)

        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Saved evaluation report to {filepath}")
