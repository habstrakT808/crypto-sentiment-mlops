"""
Train FinBERT Model
Script to train FinBERT sentiment classifier
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import argparse
from datetime import datetime

from src.models.finbert_model import FinBERTModel
from src.models.model_trainer import ModelTrainer
from src.mlflow_tracking.experiment_tracker import ExperimentTracker
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main(args):
    """Main training function"""
    logger.info("="*50)
    logger.info("FINBERT MODEL TRAINING")
    logger.info("="*50)
    
    # Load data
    logger.info(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check required columns
    required_columns = ['preprocessed_text', 'auto_label_id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"❌ Missing required columns: {missing_columns}")
        logger.error("Please run auto_label_data.py first!")
        return
    
    # Initialize model
    logger.info("\nInitializing FinBERT model...")
    model = FinBERTModel(
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        device=args.device
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
    run_name = f"finbert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"\nStarting MLflow run: {run_name}")
    
    with tracker.start_run(run_name=run_name):
        # Log parameters
        params = {
            'model_type': 'finbert_sentiment',
            'finbert_model': args.model_name,
            'max_len': args.max_len,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs,
            'device': model.device,
            'training_samples': len(df)
        }
        tracker.log_params(params)
        
        # Train model
        logger.info("\nTraining FinBERT model...")
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
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("TRAINING RESULTS")
        logger.info("="*50)
        logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"F1 (weighted):      {metrics['f1_weighted']:.4f}")
        
        logger.info("\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FinBERT model")
    
    parser.add_argument('--data_path', type=str, default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"))
    parser.add_argument('--model_name', type=str, default='ProsusAI/finbert')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    main(args)