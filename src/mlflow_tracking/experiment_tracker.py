"""
Experiment Tracker
MLflow experiment tracking wrapper
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class ExperimentTracker:
    """Track experiments with MLflow"""
    
    def __init__(self, experiment_name: str = None):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of MLflow experiment
        """
        self.experiment_name = experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        
        # Set tracking URI
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new experiment: {self.experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: str = None) -> mlflow.ActiveRun:
        """Start MLflow run"""
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics
        - Logs only scalar numeric metrics to MLflow metrics
        - Logs structured metrics (per_class, confusion_matrix, classification_report) as artifacts
        """
        flat_metrics: Dict[str, float] = {}
        per_class = metrics.get('per_class') if isinstance(metrics, dict) else None
        classification_report = metrics.get('classification_report') if isinstance(metrics, dict) else None
        confusion_matrix = metrics.get('confusion_matrix') if isinstance(metrics, dict) else None

        # Collect top-level scalar metrics
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key in ('per_class', 'classification_report', 'confusion_matrix'):
                    continue
                if isinstance(value, (int, float)):
                    flat_metrics[key] = float(value)

        # Expand per-class metrics into flat keys
        if isinstance(per_class, dict):
            for class_name, class_vals in per_class.items():
                if not isinstance(class_vals, dict):
                    continue
                for metric_name, metric_value in class_vals.items():
                    if isinstance(metric_value, (int, float)):
                        flat_metrics[f"per_class.{class_name}.{metric_name}"] = float(metric_value)

        # Log flat numeric metrics
        if flat_metrics:
            mlflow.log_metrics(flat_metrics, step=step)

        # Log structured artifacts for richer detail
        if classification_report is not None:
            mlflow.log_dict(classification_report, "classification_report.json")
        if confusion_matrix is not None:
            # Ensure serializable list form
            if hasattr(confusion_matrix, 'tolist'):
                cm_serializable = confusion_matrix.tolist()
            else:
                cm_serializable = confusion_matrix
            mlflow.log_dict({"confusion_matrix": cm_serializable}, "confusion_matrix.json")
    
    def log_model(self, model: Any, artifact_path: str, model_type: str = 'sklearn'):
        """
        Log model
        
        Args:
            model: Model to log
            artifact_path: Artifact path
            model_type: Type of model ('sklearn', 'pytorch', etc)
        """
        if model_type == 'sklearn':
            mlflow.sklearn.log_model(model, artifact_path)
        elif model_type == 'pytorch':
            mlflow.pytorch.log_model(model, artifact_path)
        else:
            logger.warning(f"Unknown model type: {model_type}")
    
    def log_artifact(self, local_path: Path):
        """Log artifact file"""
        mlflow.log_artifact(str(local_path))
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log dictionary as artifact"""
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_figure(self, figure, artifact_file: str):
        """Log matplotlib figure"""
        mlflow.log_figure(figure, artifact_file)
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
    
    def track_training(
        self,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model: Any = None,
        artifacts: Dict[str, Path] = None
    ):
        """
        Complete tracking workflow
        
        Args:
            model_name: Name of the model
            params: Model parameters
            metrics: Training metrics
            model: Trained model
            artifacts: Additional artifacts to log
        """
        with self.start_run(run_name=model_name):
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            if model is not None:
                self.log_model(model, "model")
            
            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    self.log_artifact(path)
            
            logger.info(f"Logged run for {model_name}")