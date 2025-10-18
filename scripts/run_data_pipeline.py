"""
Run Data Pipeline Manually
For testing without Airflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.reddit_collector import RedditCollector
from src.data.data_validator import DataValidator
from src.data.preprocessor import TextPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.utils.config import Config
from src.utils.logger import setup_logger
import pandas as pd
from datetime import datetime

logger = setup_logger(__name__)

def run_complete_pipeline():
    """Run complete data pipeline"""
    
    print("\n" + "="*80)
    print("CRYPTO SENTIMENT DATA PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Data Collection
    print("Step 1/5: Collecting data from Reddit...")
    print("-" * 80)
    
    collector = RedditCollector()
    df = collector.collect_from_multiple_subreddits(
        subreddits=Config.SUBREDDITS,
        limit_per_subreddit=100,
        sort_by="hot"
    )
    
    if df.empty:
        logger.error("No data collected!")
        return
    
    print(f"✅ Collected {len(df)} posts from {df['subreddit'].nunique()} subreddits\n")
    
    # Save raw data
    raw_filepath = collector.save_to_csv(df)
    
    # Step 2: Data Validation
    print("Step 2/5: Validating data quality...")
    print("-" * 80)
    
    validator = DataValidator()
    validation_summary = validator.validate_all(df)
    
    if not validation_summary['overall_passed']:
        logger.error("Data quality validation failed!")
        print(f"❌ Critical issues: {validation_summary['critical_issues']}")
        print(f"⚠️  Warnings: {validation_summary['warnings']}")
        return
    
    print(f"✅ Data quality validation passed!")
    print(f"   - Checks passed: {validation_summary['passed_checks']}/{validation_summary['total_checks']}")
    print(f"   - Warnings: {validation_summary['warnings']}\n")
    
    # Step 3: Data Preprocessing
    print("Step 3/5: Preprocessing text data...")
    print("-" * 80)
    
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True
    )
    
    df = preprocessor.process_dataframe(df, text_column='content')
    
    print(f"✅ Preprocessing complete!")
    print(f"   - Cleaned text created")
    print(f"   - Preprocessed text created")
    print(f"   - Sentiment features extracted")
    print(f"   - Text statistics computed\n")
    
    # Save preprocessed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_path = Config.PROCESSED_DATA_DIR / f"processed_posts_{timestamp}.csv"
    df.to_csv(processed_path, index=False)
    
    # Step 4: Feature Engineering
    print("Step 4/5: Engineering features...")
    print("-" * 80)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(
        df,
        include_tfidf=False,
        include_topics=False
    )
    
    print(f"✅ Feature engineering complete!")
    print(f"   - Total columns: {len(df.columns)}")
    print(f"   - Temporal features: ✓")
    print(f"   - Engagement features: ✓")
    print(f"   - Text complexity features: ✓")
    print(f"   - Sentiment derivative features: ✓")
    print(f"   - Subreddit features: ✓\n")
    
    # Save features
    features_path = Config.PROCESSED_DATA_DIR / f"features_{timestamp}.csv"
    df.to_csv(features_path, index=False)
    
    # Step 5: Summary Report
    print("Step 5/5: Generating summary report...")
    print("-" * 80)
    
    # Feature summary
    feature_summary = engineer.get_feature_importance_summary(df)
    
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Total Records: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Subreddits: {df['subreddit'].nunique()}")
    print(f"Date Range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    print(f"\nData Quality:")
    print(f"  - Completeness: {validation_summary['passed_checks']}/{validation_summary['total_checks']} checks passed")
    print(f"  - Critical Issues: {validation_summary['critical_issues']}")
    print(f"  - Warnings: {validation_summary['warnings']}")
    print(f"\nOutput Files:")
    print(f"  - Raw data: {raw_filepath}")
    print(f"  - Processed data: {processed_path}")
    print(f"  - Features: {features_path}")
    print("\nTop 10 Engineered Features:")
    print(feature_summary.head(10).to_string(index=False))
    print("="*80)
    
    print("\n✅ PIPELINE COMPLETE!\n")
    
    return df, features_path

if __name__ == "__main__":
    try:
        df, features_path = run_complete_pipeline()
        print(f"\n🎉 Success! Features saved to: {features_path}\n")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}\n")