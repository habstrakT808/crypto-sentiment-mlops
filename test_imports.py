#!/usr/bin/env python
"""
Test all imports to ensure no syntax errors
"""
import sys

def test_import(module_name, description):
    """Test importing a module"""
    try:
        __import__(module_name)
        print(f"[OK] {description}")
        return True
    except Exception as e:
        print(f"[FAILED] {description}: {e}")
        return False

def main():
    """Run all import tests"""
    print("=" * 60)
    print("TESTING ALL IMPORTS")
    print("=" * 60)
    
    tests = [
        ("src.api.models", "API Models"),
        ("src.api.dependencies", "API Dependencies"),
        ("src.api.middleware", "API Middleware"),
        ("src.api.routers.health", "Health Router"),
        ("src.api.routers.prediction", "Prediction Router"),
        ("src.api.routers.monitoring", "Monitoring Router"),
        ("src.api.main", "Main API"),
    ]
    
    results = []
    for module_name, description in tests:
        results.append(test_import(module_name, description))
    
    print("=" * 60)
    if all(results):
        print("ALL TESTS PASSED")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())
