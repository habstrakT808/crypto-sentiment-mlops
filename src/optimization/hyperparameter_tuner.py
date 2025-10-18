# File: src/optimization/hyperparameter_tuner.py
"""
Hyperparameter Tuning
Automated hyperparameter optimization using Optuna
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Callable
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HyperparameterTuner:
    """Automated hyperparameter tuning with Optuna"""
    
    def __init__(
        self,
        n_trials: int = 50,
        n_folds: int = 5,
        random_state: int = 42,
        direction: str = 'maximize'
    ):
        """
        Initialize tuner
        
        Args:
            n_trials: Number of optimization trials
            n_folds: Number of cross-validation folds
            random_state: Random seed
            direction: Optimization direction ('maximize' or 'minimize')
        """
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.random_state = random_state
        self.direction = direction
        self.study = None
        self.best_params = None
        
        logger.info(f"HyperparameterTuner initialized: {n_trials} trials, {n_folds} folds")
    
    def _objective_lightgbm(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> float:
        """Objective function for LightGBM"""
        # Filter only numeric columns for LightGBM
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_columns]
        
        # Define hyperparameter search space
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': self.random_state,
            
            # Hyperparameters to tune
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_numeric, y)):
            X_train_fold = X_numeric.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_numeric.iloc[val_idx]
            y_val_fold = y[val_idx]
            
            # Train model
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # Predict
            y_pred = model.predict(X_val_fold)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            # Calculate F1 score
            fold_score = f1_score(y_val_fold, y_pred_class, average='weighted')
            cv_scores.append(fold_score)
        
        return np.mean(cv_scores)
    
    def _objective_deberta(
        self,
        trial: optuna.Trial,
        X: pd.Series,
        y: np.ndarray
    ) -> float:
        """Objective function for DeBERTa"""
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'num_epochs': trial.suggest_int('num_epochs', 3, 7),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        }
        
        # Import model
        from src.models.deberta_model import DeBERTaV3Model
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y[val_idx]
            
            # Initialize model
            model = DeBERTaV3Model(
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                num_epochs=params['num_epochs'],
                warmup_ratio=params['warmup_ratio'],
                weight_decay=params['weight_decay'],
                focal_gamma=params['focal_gamma']
            )
            
            # Train
            model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
            
            # Predict
            y_pred = model.predict(X_val_fold)
            
            # Calculate F1 score
            fold_score = f1_score(y_val_fold, y_pred, average='weighted')
            cv_scores.append(fold_score)
        
        return np.mean(cv_scores)
    
    def tune_lightgbm(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        study_name: str = 'lightgbm_tuning'
    ) -> Dict[str, Any]:
        """
        Tune LightGBM hyperparameters
        
        Args:
            X: Feature matrix
            y: Labels
            study_name: Name of the study
            
        Returns:
            Best parameters
        """
        logger.info(f"Starting LightGBM hyperparameter tuning ({self.n_trials} trials)...")
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=study_name
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective_lightgbm(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best F1 Score: {self.study.best_value:.4f}")
        logger.info(f"Best Parameters: {self.best_params}")
        
        return self.best_params
    
    def tune_deberta(
        self,
        X: pd.Series,
        y: np.ndarray,
        study_name: str = 'deberta_tuning'
    ) -> Dict[str, Any]:
        """
        Tune DeBERTa hyperparameters
        
        Args:
            X: Text series
            y: Labels
            study_name: Name of the study
            
        Returns:
            Best parameters
        """
        logger.info(f"Starting DeBERTa hyperparameter tuning ({self.n_trials} trials)...")
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            study_name=study_name
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective_deberta(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        logger.info(f"Optimization complete!")
        logger.info(f"Best F1 Score: {self.study.best_value:.4f}")
        logger.info(f"Best Parameters: {self.best_params}")
        
        return self.best_params
    
    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history"""
        if self.study is None:
            raise ValueError("No study available. Run tuning first.")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization history saved to {save_path}")
        
        plt.close()
    
    def plot_param_importances(self, save_path: str = None):
        """Plot parameter importances"""
        if self.study is None:
            raise ValueError("No study available. Run tuning first.")
        
        import matplotlib.pyplot as plt
        
        fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Parameter importances saved to {save_path}")
        
        plt.close()