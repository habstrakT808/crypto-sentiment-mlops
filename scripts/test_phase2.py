"""
Test Phase 2 Components
Test each component individually
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_validator import DataValidator
from src.data.preprocessor import TextPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.utils.config import Config
import pandas as pd

def test_data_validator():
    """Test data validator"""
    print("\n" + "="*50)
    print("TESTING DATA VALIDATOR")
    print("="*50)
    
    # Load sample data
    raw_files = list(Config.RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print("❌ No raw data files found. Run data collection first.")
        return False
    
    df = pd.read_csv(raw_files[0])
    print(f"Loaded {len(df)} records from {raw_files[0].name}")
    
    # Validate
    validator = DataValidator()
    summary = validator.validate_all(df)
    
    print(f"\n✅ Validation complete!")
    print(f"Overall passed: {summary['overall_passed']}")
    print(f"Checks: {summary['passed_checks']}/{summary['total_checks']} passed")
    print(f"Critical issues: {summary['critical_issues']}")
    print(f"Warnings: {summary['warnings']}")
    
    return summary['overall_passed']

def test_preprocessor():
    """Test text preprocessor"""
    print("\n" + "="*50)
    print("TESTING TEXT PREPROCESSOR")
    print("="*50)
    
    # Sample texts
    test_texts = [
        "Bitcoin is going to the MOON! 🚀 Check out https://example.com #crypto",
        "I'm really worried about the crash. What should I do?",
        "Ethereum's technology is revolutionary for DeFi applications"
    ]
    
    preprocessor = TextPreprocessor()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {text}")
        
        cleaned = preprocessor.clean_text(text)
        print(f"Cleaned: {cleaned}")
        
        preprocessed = preprocessor.preprocess(cleaned)
        print(f"Preprocessed: {preprocessed}")
        
        sentiment = preprocessor.extract_sentiment_features(cleaned)
        print(f"Sentiment: {sentiment}")
        
        stats = preprocessor.extract_text_statistics(cleaned)
        print(f"Stats: {stats}")
    
    print("\n✅ Preprocessor test complete!")
    return True

def test_feature_engineer():
    """Test feature engineer"""
    print("\n" + "="*50)
    print("TESTING FEATURE ENGINEER")
    print("="*50)
    
    # Load processed data
    processed_files = list(Config.PROCESSED_DATA_DIR.glob("processed_*.csv"))
    if not processed_files:
        print("❌ No processed data files found. Run preprocessing first.")
        return False
    
    df = pd.read_csv(processed_files[0])
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    
    print(f"Loaded {len(df)} records")
    print(f"Initial columns: {len(df.columns)}")
    
    # Engineer features
    engineer = FeatureEngineer()
    df = engineer.engineer_all_features(df)
    
    print(f"Final columns: {len(df.columns)}")
    print(f"Features added: {len(df.columns) - len(processed_files)}")
    
    # Show sample features
    feature_summary = engineer.get_feature_importance_summary(df)
    print("\nFeature Summary:")
    print(feature_summary.head(15).to_string(index=False))
    
    print("\n✅ Feature engineering test complete!")
    return True

def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n" + "="*70)
    print("PHASE 2 COMPONENT TESTING")
    print("="*70)
    
    results = {
        'Data Validator': test_data_validator(),
        'Text Preprocessor': test_preprocessor(),
        'Feature Engineer': test_feature_engineer()
    }
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Phase 2 is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*70 + "\n")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()