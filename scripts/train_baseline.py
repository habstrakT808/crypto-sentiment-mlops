"""
Train Baseline Model
Script to train baseline Logistic Regression model
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import argparse
from datetime import datetime

from src.models.baseline_model import BaselineModel
from src.models.model_trainer import ModelTrainer
from src.mlflow_tracking.experiment_tracker import ExperimentTracker
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main(args):
    """Main training function"""
    logger.info("="*50)
    logger.info("BASELINE MODEL TRAINING")
    logger.info("="*50)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check if required columns exist
    required_columns = ['preprocessed_text', 'auto_label_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        logger.error("Please run auto_label_data.py first!")
        return
    
    # Filter high confidence samples if specified
    if args.use_high_confidence:
        logger.info("Filtering high confidence samples...")
        initial_count = len(df)
        df = df[
            (df['label_confidence'] >= args.min_confidence) &
            (df['label_agreement'] >= args.min_agreement)
        ]
        logger.info(f"Filtered from {initial_count} to {len(df)} samples ({len(df)/initial_count*100:.1f}% retained)")
        
        if len(df) < 50:
            logger.error("❌ Too few samples after filtering. Lower confidence thresholds or use more data.")
            return
    
    # Log label distribution
    logger.info("\nLabel distribution:")
    label_counts = df['auto_label_id'].value_counts().sort_index()
    for label_id, count in label_counts.items():
        label_name = ['negative', 'neutral', 'positive'][label_id]
        logger.info(f"  {label_name} ({label_id}): {count} ({count/len(df)*100:.1f}%)")
    
    # Initialize model
    logger.info("\nInitializing baseline model...")
    model = BaselineModel(
        max_features=args.max_features,
        ngram_range=(1, args.max_ngram),
        C=args.C,
        max_iter=args.max_iter
    )
    
    # Initialize trainer
    trainer = ModelTrainer(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    # Initialize MLflow tracker
    tracker = ExperimentTracker()
    
    # Train and evaluate
    run_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"\nStarting MLflow run: {run_name}")
    
    with tracker.start_run(run_name=run_name):
        # Log parameters
        params = {
            'model_type': 'baseline_logistic_regression',
            'max_features': args.max_features,
            'ngram_range': f"(1, {args.max_ngram})",
            'C': args.C,
            'max_iter': args.max_iter,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'use_high_confidence': args.use_high_confidence,
            'min_confidence': args.min_confidence if args.use_high_confidence else None,
            'min_agreement': args.min_agreement if args.use_high_confidence else None,
            'training_samples': len(df)
        }
        tracker.log_params(params)
        
        # Train model
        logger.info("\nTraining model...")
        results = trainer.train_and_evaluate(
            model=model,
            df=df,
            text_column='preprocessed_text',
            label_column='auto_label_id',
            save_model=True
        )
        
        # Log metrics
        metrics = results['evaluation']['metrics']
        tracker.log_metrics(metrics)
        
        # Log model
        logger.info("Logging model to MLflow...")
        tracker.log_model(results['model'].model, "model", model_type='sklearn')
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS")
        logger.info("="*50)
        logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"F1 (macro):         {metrics['f1_macro']:.4f}")
        logger.info(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
        logger.info(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        
        # Per-class metrics
        logger.info("\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            logger.info(f"  {class_name:10s}: P={class_metrics['precision']:.4f}, "
                       f"R={class_metrics['recall']:.4f}, F1={class_metrics['f1']:.4f}")
        
        # Visualize results
        if args.visualize:
            logger.info("\nGenerating visualizations...")
            evaluator = ModelEvaluator()
            
            # Plot confusion matrix
            cm = results['evaluation']['metrics']['confusion_matrix']
            cm_path = Config.MODELS_DIR / f"baseline_confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            evaluator.plot_confusion_matrix(
                cm,
                class_names=['Negative', 'Neutral', 'Positive'],
                save_path=cm_path
            )
            logger.info(f"✅ Confusion matrix saved to {cm_path}")
            
            # Log to MLflow
            tracker.log_artifact(cm_path)
        
        # Get feature importance
        logger.info("\nTop features per class:")
        feature_importance = results['model'].get_feature_importance(top_n=10)
        for class_name, features in feature_importance.items():
            logger.info(f"\n  {class_name.upper()}:")
            for feature, coef in features[:5]:
                logger.info(f"    {feature:20s}: {coef:+.4f}")
        
        logger.info("\n" + "="*50)
        logger.info("✅ Training complete!")
        logger.info("="*50)
        logger.info(f"\nModel saved to: {Config.MODELS_DIR / 'baseline_logistic_regression_latest.pkl'}")
        logger.info(f"MLflow run: {run_name}")
        logger.info(f"View results: mlflow ui --port 5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline model")
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"),
        help="Path to labeled data"
    )
    
    # Model arguments
    parser.add_argument(
        '--max_features',
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features"
    )
    parser.add_argument(
        '--max_ngram',
        type=int,
        default=2,
        help="Maximum n-gram size"
    )
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help="Regularization parameter"
    )
    parser.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help="Maximum iterations"
    )
    
    # Training arguments
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.15,
        help="Test set size"
    )
    parser.add_argument(
        '--val_size',
        type=float,
        default=0.15,
        help="Validation set size"
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help="Random seed"
    )
    
    # Filtering arguments
    parser.add_argument(
        '--use_high_confidence',
        action='store_true',
        help="Use only high confidence samples"
    )
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=0.6,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        '--min_agreement',
        type=float,
        default=0.67,
        help="Minimum agreement threshold"
    )
    
    # Other arguments
    parser.add_argument(
        '--visualize',
        action='store_true',
        help="Generate visualizations"
    )
    
    args = parser.parse_args()
    main(args)