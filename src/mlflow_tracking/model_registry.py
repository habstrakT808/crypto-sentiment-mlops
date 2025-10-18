"""
Model Registry
MLflow model registry wrapper
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, List
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


class ModelRegistry:
    """Manage models in MLflow registry"""
    
    def __init__(self):
        """Initialize model registry"""
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> str:
        """
        Register model
        
        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            artifact_path: Path to model artifact
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        try:
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            logger.info(f"Registered {model_name} version {version}")
            return version
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str
    ):
        """
        Transition model to different stage
        
        Args:
            model_name: Name of model
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Transitioned {model_name} v{version} to {stage}")
    
    def get_latest_model_version(
        self,
        model_name: str,
        stage: str = "Production"
    ) -> str:
        """Get latest model version for stage"""
        versions = self.client.get_latest_versions(model_name, stages=[stage])
        if versions:
            return versions[0].version
        return None
    
    def list_registered_models(self) -> pd.DataFrame:
        """List all registered models"""
        models = self.client.list_registered_models()
        
        data = []
        for model in models:
            data.append({
                'name': model.name,
                'latest_version': model.latest_versions[0].version if model.latest_versions else None,
                'creation_time': model.creation_timestamp
            })
        
        return pd.DataFrame(data)