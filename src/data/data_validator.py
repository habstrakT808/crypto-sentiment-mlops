"""
Data Validation Module
Comprehensive data quality checks and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import re
from dataclasses import dataclass

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

@dataclass
class ValidationResult:
    """Data validation result"""
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    message: str
    severity: str  # 'critical', 'warning', 'info'

class DataValidator:
    """Validate data quality and completeness"""
    
    def __init__(self):
        """Initialize validator with quality thresholds"""
        self.thresholds = {
            'completeness': 0.95,  # 95% non-null
            'duplicate_rate': 0.05,  # Max 5% duplicates
            'min_text_length': 10,
            'max_text_length': 10000,
            'min_score': -1000,
            'max_score': 100000,
            'max_data_age_hours': 48
        }
        
        self.validation_results: List[ValidationResult] = []
    
    def validate_completeness(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check data completeness
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object
        """
        required_columns = ['post_id', 'title', 'content', 'created_utc', 'subreddit']
        
        try:
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return ValidationResult(
                    passed=False,
                    metric_name='column_completeness',
                    metric_value=0.0,
                    threshold=1.0,
                    message=f"Missing required columns: {missing_columns}",
                    severity='critical'
                )
            
            # Check non-null percentage for each required column
            completeness_scores = {}
            for col in required_columns:
                non_null_pct = df[col].notna().sum() / len(df)
                completeness_scores[col] = non_null_pct
            
            # Overall completeness
            avg_completeness = np.mean(list(completeness_scores.values()))
            
            passed = avg_completeness >= self.thresholds['completeness']
            severity = 'info' if passed else ('warning' if avg_completeness > 0.90 else 'critical')
            
            result = ValidationResult(
                passed=passed,
                metric_name='data_completeness',
                metric_value=avg_completeness,
                threshold=self.thresholds['completeness'],
                message=f"Data completeness: {avg_completeness:.2%}. Details: {completeness_scores}",
                severity=severity
            )
            
            self.validation_results.append(result)
            logger.info(result.message)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in completeness validation: {e}")
            return ValidationResult(
                passed=False,
                metric_name='data_completeness',
                metric_value=0.0,
                threshold=self.thresholds['completeness'],
                message=f"Validation error: {str(e)}",
                severity='critical'
            )
    
    def validate_duplicates(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check for duplicate records
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object
        """
        try:
            total_records = len(df)
            
            # Check duplicates based on post_id
            if 'post_id' in df.columns:
                duplicate_count = df['post_id'].duplicated().sum()
            else:
                duplicate_count = df.duplicated().sum()
            
            duplicate_rate = duplicate_count / total_records if total_records > 0 else 0
            
            passed = duplicate_rate <= self.thresholds['duplicate_rate']
            severity = 'info' if passed else ('warning' if duplicate_rate < 0.10 else 'critical')
            
            result = ValidationResult(
                passed=passed,
                metric_name='duplicate_rate',
                metric_value=duplicate_rate,
                threshold=self.thresholds['duplicate_rate'],
                message=f"Duplicate rate: {duplicate_rate:.2%} ({duplicate_count}/{total_records} records)",
                severity=severity
            )
            
            self.validation_results.append(result)
            logger.info(result.message)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in duplicate validation: {e}")
            return ValidationResult(
                passed=False,
                metric_name='duplicate_rate',
                metric_value=1.0,
                threshold=self.thresholds['duplicate_rate'],
                message=f"Validation error: {str(e)}",
                severity='critical'
            )
    
    def validate_text_quality(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate text content quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object
        """
        try:
            # Combine title and content for analysis
            df['full_text'] = df['title'].fillna('') + ' ' + df['content'].fillna('')
            df['text_length'] = df['full_text'].str.len()
            
            # Check text length distribution
            too_short = (df['text_length'] < self.thresholds['min_text_length']).sum()
            too_long = (df['text_length'] > self.thresholds['max_text_length']).sum()
            valid_length = len(df) - too_short - too_long
            
            valid_rate = valid_length / len(df) if len(df) > 0 else 0
            
            passed = valid_rate >= 0.90  # 90% should have valid length
            severity = 'info' if passed else ('warning' if valid_rate > 0.80 else 'critical')
            
            result = ValidationResult(
                passed=passed,
                metric_name='text_quality',
                metric_value=valid_rate,
                threshold=0.90,
                message=f"Text quality: {valid_rate:.2%} valid. Too short: {too_short}, Too long: {too_long}",
                severity=severity
            )
            
            self.validation_results.append(result)
            logger.info(result.message)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in text quality validation: {e}")
            return ValidationResult(
                passed=False,
                metric_name='text_quality',
                metric_value=0.0,
                threshold=0.90,
                message=f"Validation error: {str(e)}",
                severity='critical'
            )
    
    def validate_data_freshness(self, df: pd.DataFrame) -> ValidationResult:
        """
        Check if data is recent enough
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object
        """
        try:
            if 'created_utc' not in df.columns:
                return ValidationResult(
                    passed=False,
                    metric_name='data_freshness',
                    metric_value=0.0,
                    threshold=self.thresholds['max_data_age_hours'],
                    message="Column 'created_utc' not found",
                    severity='critical'
                )
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['created_utc']):
                df['created_utc'] = pd.to_datetime(df['created_utc'])
            
            # Calculate data age
            now = datetime.now()
            df['data_age_hours'] = (now - df['created_utc']).dt.total_seconds() / 3600
            
            # Check percentage of fresh data
            fresh_data = (df['data_age_hours'] <= self.thresholds['max_data_age_hours']).sum()
            fresh_rate = fresh_data / len(df) if len(df) > 0 else 0
            
            avg_age_hours = df['data_age_hours'].mean()
            
            passed = fresh_rate >= 0.50  # At least 50% should be fresh
            severity = 'info' if passed else 'warning'
            
            result = ValidationResult(
                passed=passed,
                metric_name='data_freshness',
                metric_value=avg_age_hours,
                threshold=self.thresholds['max_data_age_hours'],
                message=f"Data freshness: {fresh_rate:.2%} within {self.thresholds['max_data_age_hours']}h. Avg age: {avg_age_hours:.1f}h",
                severity=severity
            )
            
            self.validation_results.append(result)
            logger.info(result.message)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in freshness validation: {e}")
            return ValidationResult(
                passed=False,
                metric_name='data_freshness',
                metric_value=999.0,
                threshold=self.thresholds['max_data_age_hours'],
                message=f"Validation error: {str(e)}",
                severity='critical'
            )
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data schema and types
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object
        """
        try:
            expected_schema = {
                'post_id': 'object',
                'title': 'object',
                'content': 'object',
                'author': 'object',
                'score': 'int64',
                'num_comments': 'int64',
                'subreddit': 'object'
            }
            
            schema_issues = []
            
            for col, expected_type in expected_schema.items():
                if col not in df.columns:
                    schema_issues.append(f"Missing column: {col}")
                elif not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                    # Try to convert
                    try:
                        if expected_type == 'int64':
                            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                        elif expected_type == 'object':
                            df[col] = df[col].astype('object')
                    except:
                        schema_issues.append(f"Column {col} has type {df[col].dtype}, expected {expected_type}")
            
            passed = len(schema_issues) == 0
            severity = 'info' if passed else 'warning'
            
            result = ValidationResult(
                passed=passed,
                metric_name='schema_validation',
                metric_value=1.0 if passed else 0.0,
                threshold=1.0,
                message=f"Schema validation: {'Passed' if passed else 'Issues found: ' + ', '.join(schema_issues)}",
                severity=severity
            )
            
            self.validation_results.append(result)
            logger.info(result.message)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in schema validation: {e}")
            return ValidationResult(
                passed=False,
                metric_name='schema_validation',
                metric_value=0.0,
                threshold=1.0,
                message=f"Validation error: {str(e)}",
                severity='critical'
            )
    
    def validate_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all validation checks
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation summary
        """
        logger.info(f"Starting comprehensive validation on {len(df)} records")
        
        self.validation_results = []
        
        # Run all validations
        self.validate_completeness(df)
        self.validate_duplicates(df)
        self.validate_text_quality(df)
        self.validate_data_freshness(df)
        self.validate_schema(df)
        
        # Summarize results
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        critical_issues = [r for r in self.validation_results if r.severity == 'critical' and not r.passed]
        warnings = [r for r in self.validation_results if r.severity == 'warning' and not r.passed]
        
        overall_passed = len(critical_issues) == 0
        
        summary = {
            'overall_passed': overall_passed,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'critical_issues': len(critical_issues),
            'warnings': len(warnings),
            'validation_results': self.validation_results,
            'timestamp': datetime.now()
        }
        
        # Log summary
        logger.info(f"Validation complete: {passed_checks}/{total_checks} checks passed")
        if critical_issues:
            logger.error(f"Critical issues found: {len(critical_issues)}")
            for issue in critical_issues:
                logger.error(f"  - {issue.message}")
        
        if warnings:
            logger.warning(f"Warnings: {len(warnings)}")
            for warning in warnings:
                logger.warning(f"  - {warning.message}")
        
        return summary
    
    def save_validation_report(self, summary: Dict[str, Any], output_path: str = None):
        """
        Save validation report to file
        
        Args:
            summary: Validation summary dictionary
            output_path: Optional output file path
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Config.LOGS_DIR / f"validation_report_{timestamp}.json"
            
            import json
            
            # Convert ValidationResult objects to dict
            report = {
                'summary': {
                    'overall_passed': summary['overall_passed'],
                    'total_checks': summary['total_checks'],
                    'passed_checks': summary['passed_checks'],
                    'failed_checks': summary['failed_checks'],
                    'critical_issues': summary['critical_issues'],
                    'warnings': summary['warnings'],
                    'timestamp': summary['timestamp'].isoformat()
                },
                'detailed_results': [
                    {
                        'metric_name': r.metric_name,
                        'passed': r.passed,
                        'metric_value': r.metric_value,
                        'threshold': r.threshold,
                        'message': r.message,
                        'severity': r.severity
                    }
                    for r in summary['validation_results']
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation report: {e}")