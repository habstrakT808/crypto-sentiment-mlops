"""
Auto Label Data
Script to automatically label collected data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import argparse
from datetime import datetime

from src.models.labeling.auto_labeler import AutoLabeler
from src.models.labeling.confidence_filter import ConfidenceFilter
from src.models.labeling.label_validator import LabelValidator
from src.data.preprocessor import TextPreprocessor
from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)


def main(args):
    """Main labeling function"""
    logger.info("="*50)
    logger.info("AUTO-LABELING PIPELINE")
    logger.info("="*50)
    
    # Load data
    logger.info(f"Loading data from {args.input_path}...")
    df = pd.read_csv(args.input_path)
    logger.info(f"Loaded {len(df)} samples")
    
    # Check available columns
    logger.info(f"Available columns: {df.columns.tolist()}")
    
    # Combine title and content for preprocessing
    logger.info("Combining title and content...")
    df['full_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
    
    # Preprocess text
    logger.info("Preprocessing text...")
    preprocessor = TextPreprocessor()
    
    # Create preprocessed_text column
    df['preprocessed_text'] = df['full_text'].apply(preprocessor.preprocess)
    
    logger.info(f"Preprocessing complete. Sample:")
    logger.info(f"Original: {df['full_text'].iloc[0][:100]}...")
    logger.info(f"Preprocessed: {df['preprocessed_text'].iloc[0][:100]}...")
    
    # Initialize auto-labeler
    logger.info("\nInitializing auto-labeler...")
    labeler = AutoLabeler(device=args.device)
    
    # Label data
    logger.info("Starting auto-labeling...")
    df = labeler.label_dataframe(
        df,
        text_column='preprocessed_text',
        batch_size=args.batch_size
    )
    
    # Get labeling stats
    stats = labeler.get_labeling_stats(df)
    logger.info("\nLabeling Statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Average confidence: {stats['avg_confidence']:.4f}")
    logger.info(f"  Average agreement: {stats['avg_agreement']:.4f}")
    logger.info(f"  Label distribution:")
    for label, count in stats['label_distribution'].items():
        logger.info(f"    {label}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    # Validate labels
    if args.validate:
        logger.info("\nValidating labels...")
        validator = LabelValidator()
        validation_report = validator.comprehensive_validation(df)
        logger.info(f"Validation status: {validation_report['overall_status']}")
        
        if validation_report['overall_status'] == 'NEEDS_REVIEW':
            logger.warning("⚠️  Validation needs review. Check logs for details.")
    
    # Filter by confidence
    if args.filter_confidence:
        logger.info("\nFiltering by confidence...")
        conf_filter = ConfidenceFilter(
            min_confidence=args.min_confidence,
            min_agreement=args.min_agreement,
            use_both_criteria=True
        )
        
        high_conf_df, low_conf_df = conf_filter.filter(df)
        
        # Save high confidence data
        high_conf_path = args.output_path.replace('.csv', '_high_confidence.csv')
        high_conf_df.to_csv(high_conf_path, index=False)
        logger.info(f"✅ High confidence data saved to {high_conf_path}")
        
        # Save low confidence data for review
        low_conf_path = args.output_path.replace('.csv', '_low_confidence.csv')
        low_conf_df.to_csv(low_conf_path, index=False)
        logger.info(f"✅ Low confidence data saved to {low_conf_path}")
        
        # Use high confidence for main output
        df_to_save = high_conf_df
    else:
        df_to_save = df
    
    # Save labeled data
    logger.info(f"\nSaving labeled data to {args.output_path}...")
    df_to_save.to_csv(args.output_path, index=False)
    logger.info(f"✅ Saved {len(df_to_save)} labeled samples")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("AUTO-LABELING SUMMARY")
    logger.info("="*50)
    logger.info(f"Input samples: {len(df)}")
    logger.info(f"Output samples: {len(df_to_save)}")
    logger.info(f"Retention rate: {len(df_to_save)/len(df)*100:.1f}%")
    logger.info(f"Average confidence: {df_to_save['label_confidence'].mean():.4f}")
    logger.info(f"Average agreement: {df_to_save['label_agreement'].mean():.4f}")
    logger.info("\nLabel Distribution:")
    for label, count in df_to_save['auto_label'].value_counts().items():
        logger.info(f"  {label}: {count} ({count/len(df_to_save)*100:.1f}%)")
    logger.info("="*50)
    logger.info("✅ Auto-labeling complete!")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-label data")
    
    parser.add_argument(
        '--input_path',
        type=str,
        default=str(Config.RAW_DATA_DIR / "reddit_posts_20251010_150745.csv"),
        help="Path to input data"
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=str(Config.PROCESSED_DATA_DIR / "labeled_data.csv"),
        help="Path to save labeled data"
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help="Validate labels after labeling"
    )
    parser.add_argument(
        '--filter_confidence',
        action='store_true',
        help="Filter by confidence scores"
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
    
    args = parser.parse_args()
    
    # Create output directory if not exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)