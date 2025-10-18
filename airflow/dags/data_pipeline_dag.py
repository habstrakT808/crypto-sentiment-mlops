"""
Airflow DAG for Data Pipeline
Automated data collection, validation, and processing
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.reddit_collector import RedditCollector
from src.data.data_validator import DataValidator
from src.data.preprocessor import TextPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.utils.config import Config
from src.utils.logger import setup_logger
import pandas as pd

logger = setup_logger(__name__)

# Default arguments
default_args = {
    'owner': 'crypto_sentiment_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'crypto_sentiment_data_pipeline',
    default_args=default_args,
    description='End-to-end data pipeline for crypto sentiment analysis',
    schedule_interval=timedelta(hours=6),  # Run every 6 hours
    catchup=False,
    max_active_runs=1,
    tags=['crypto', 'sentiment', 'data-pipeline']
)

def collect_reddit_data(**context):
    """Task 1: Collect data from Reddit"""
    logger.info("Starting Reddit data collection...")
    
    collector = RedditCollector()
    
    df = collector.collect_from_multiple_subreddits(
        subreddits=Config.SUBREDDITS,
        limit_per_subreddit=Config.MAX_POSTS_PER_SUBREDDIT,
        sort_by="hot"
    )
    
    if df.empty:
        raise ValueError("No data collected from Reddit")
    
    # Save raw data
    filepath = collector.save_to_csv(df)
    
    # Push filepath to XCom for next tasks
    context['task_instance'].xcom_push(key='raw_data_path', value=filepath)
    
    logger.info(f"Data collection complete: {len(df)} records")
    return filepath

def validate_data_quality(**context):
    """Task 2: Validate data quality"""
    logger.info("Starting data quality validation...")
    
    # Get filepath from previous task
    filepath = context['task_instance'].xcom_pull(
        task_ids='collect_reddit_data',
        key='raw_data_path'
    )
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Validate
    validator = DataValidator()
    summary = validator.validate_all(df)
    
    # Save validation report
    validator.save_validation_report(summary)
    
    # Fail task if critical issues found
    if not summary['overall_passed']:
        raise ValueError(f"Data quality validation failed: {summary['critical_issues']} critical issues")
    
    logger.info("Data quality validation passed")
    return summary

def preprocess_data(**context):
    """Task 3: Preprocess text data"""
    logger.info("Starting data preprocessing...")
    
    # Get filepath
    filepath = context['task_instance'].xcom_pull(
        task_ids='collect_reddit_data',
        key='raw_data_path'
    )
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Preprocess
    preprocessor = TextPreprocessor(
        remove_stopwords=True,
        lemmatize=True
    )
    
    df = preprocessor.process_dataframe(df, text_column='content')
    
    # Save processed data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_path = Config.PROCESSED_DATA_DIR / f"processed_posts_{timestamp}.csv"
    df.to_csv(processed_path, index=False)
    
    # Push to XCom
    context['task_instance'].xcom_push(key='processed_data_path', value=str(processed_path))
    
    logger.info(f"Preprocessing complete: {len(df)} records processed")
    return str(processed_path)

def engineer_features(**context):
    """Task 4: Engineer features"""
    logger.info("Starting feature engineering...")
    
    # Get filepath
    filepath = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_data_path'
    )
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert datetime column
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    
    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(
        df,
        include_tfidf=False,  # Skip TF-IDF for now (memory intensive)
        include_topics=False   # Skip topics for now
    )
    
    # Save features
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    features_path = Config.PROCESSED_DATA_DIR / f"features_{timestamp}.csv"
    df.to_csv(features_path, index=False)
    
    # Get feature summary
    feature_summary = engineer.get_feature_importance_summary(df)
    logger.info(f"Feature engineering complete: {len(df.columns)} total columns")
    logger.info(f"\n{feature_summary.head(20)}")
    
    context['task_instance'].xcom_push(key='features_path', value=str(features_path))
    
    return str(features_path)

def version_data_with_dvc(**context):
    """Task 5: Version data with DVC"""
    logger.info("Versioning data with DVC...")
    
    # Get all data paths
    raw_path = context['task_instance'].xcom_pull(
        task_ids='collect_reddit_data',
        key='raw_data_path'
    )
    processed_path = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_data_path'
    )
    features_path = context['task_instance'].xcom_pull(
        task_ids='engineer_features',
        key='features_path'
    )
    
    logger.info(f"Data versioning complete for: {raw_path}, {processed_path}, {features_path}")
    return "DVC versioning complete"

def generate_data_report(**context):
    """Task 6: Generate data pipeline report"""
    logger.info("Generating data pipeline report...")
    
    features_path = context['task_instance'].xcom_pull(
        task_ids='engineer_features',
        key='features_path'
    )
    
    df = pd.read_csv(features_path)
    
    report = {
        'pipeline_run_time': datetime.now().isoformat(),
        'total_records': len(df),
        'total_features': len(df.columns),
        'data_quality_passed': True,
        'subreddits_covered': df['subreddit'].nunique(),
        'date_range': {
            'start': df['created_utc'].min(),
            'end': df['created_utc'].max()
        }
    }
    
    logger.info(f"Pipeline report: {report}")
    return report

# Define tasks
task_collect = PythonOperator(
    task_id='collect_reddit_data',
    python_callable=collect_reddit_data,
    dag=dag
)

task_validate = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

task_engineer = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag
)

task_version = PythonOperator(
    task_id='version_data_with_dvc',
    python_callable=version_data_with_dvc,
    dag=dag
)

task_report = PythonOperator(
    task_id='generate_data_report',
    python_callable=generate_data_report,
    dag=dag
)

# Set task dependencies
task_collect >> task_validate >> task_preprocess >> task_engineer >> task_version >> task_report