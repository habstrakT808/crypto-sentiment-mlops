# File: src/monitoring/drift_detector.py
"""
🚨 Data Drift Detection System
Automated monitoring for model performance degradation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from scipy import stats
import joblib
from dataclasses import dataclass

from src.utils.logger import setup_logger
from src.utils.config import Config

logger = setup_logger(__name__)

@dataclass
class DriftAlert:
    """Data drift alert"""
    timestamp: datetime
    feature_name: str
    drift_score: float
    threshold: float
    severity: str  # 'warning', 'critical'
    message: str

class DataDriftDetector:
    """Advanced data drift detection system"""
    
    def __init__(self, reference_data_path: str = None):
        """Initialize drift detector"""
        self.reference_data_path = reference_data_path
        self.reference_stats = {}
        self.drift_thresholds = {
            'ks_test': 0.05,      # Kolmogorov-Smirnov test
            'psi': 0.1,           # Population Stability Index
            'js_divergence': 0.1   # Jensen-Shannon divergence
        }
        self.alerts = []
        
        if reference_data_path:
            self.load_reference_data()
        
        logger.info("DataDriftDetector initialized")
    
    def load_reference_data(self):
        """Load reference data statistics"""
        try:
            ref_data = pd.read_csv(self.reference_data_path)
            
            # Calculate reference statistics
            numeric_cols = ref_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                self.reference_stats[col] = {
                    'mean': ref_data[col].mean(),
                    'std': ref_data[col].std(),
                    'min': ref_data[col].min(),
                    'max': ref_data[col].max(),
                    'distribution': ref_data[col].values
                }
            
            logger.info(f"Reference statistics loaded for {len(numeric_cols)} features")
            
        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
    
    def detect_drift(self, new_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect data drift in new data"""
        alerts = []
        
        if not self.reference_stats:
            logger.warning("No reference data loaded, skipping drift detection")
            return alerts
        
        logger.info(f"Detecting drift in {len(new_data)} new samples...")
        
        for feature, ref_stats in self.reference_stats.items():
            if feature not in new_data.columns:
                continue
            
            new_values = new_data[feature].dropna().values
            ref_values = ref_stats['distribution']
            
            if len(new_values) < 10:  # Need minimum samples
                continue
            
            # KS Test
            ks_stat, ks_p_value = stats.ks_2samp(ref_values, new_values)
            
            if ks_p_value < self.drift_thresholds['ks_test']:
                severity = 'critical' if ks_p_value < 0.01 else 'warning'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    feature_name=feature,
                    drift_score=ks_stat,
                    threshold=self.drift_thresholds['ks_test'],
                    severity=severity,
                    message=f"KS test detected drift in {feature} (p-value: {ks_p_value:.4f})"
                ))
            
            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_values, new_values)
            
            if psi_score > self.drift_thresholds['psi']:
                severity = 'critical' if psi_score > 0.25 else 'warning'
                alerts.append(DriftAlert(
                    timestamp=datetime.now(),
                    feature_name=feature,
                    drift_score=psi_score,
                    threshold=self.drift_thresholds['psi'],
                    severity=severity,
                    message=f"PSI detected drift in {feature} (score: {psi_score:.4f})"
                ))
        
        self.alerts.extend(alerts)
        
        if alerts:
            logger.warning(f"🚨 Detected {len(alerts)} drift alerts!")
            for alert in alerts:
                logger.warning(f"   {alert.severity.upper()}: {alert.message}")
        else:
            logger.info("✅ No significant drift detected")
        
        return alerts
    
    def _calculate_psi(self, reference: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference, bins=bins)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            new_counts, _ = np.histogram(new, bins=bin_edges)
            
            # Convert to percentages
            ref_pct = ref_counts / len(reference)
            new_pct = new_counts / len(new)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            new_pct = np.where(new_pct == 0, 0.0001, new_pct)
            
            # Calculate PSI
            psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
            
            return abs(psi)
            
        except Exception as e:
            logger.warning(f"PSI calculation error: {e}")
            return 0.0
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift report"""
        recent_alerts = [
            alert for alert in self.alerts 
            if alert.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_alerts_24h": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == 'critical']),
            "warning_alerts": len([a for a in recent_alerts if a.severity == 'warning']),
            "alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "feature": alert.feature_name,
                    "score": alert.drift_score,
                    "severity": alert.severity,
                    "message": alert.message
                }
                for alert in recent_alerts
            ]
        }