"""
📊 EVIDENTLY AI INTEGRATION
Model monitoring e data drift detection
"""
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
from loguru import logger

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("evidently not installed")


class ModelMonitor:
    """
    Monitor de modelos ML usando Evidently AI
    """
    
    def __init__(self, reference_data: pd.DataFrame = None):
        if not EVIDENTLY_AVAILABLE:
            self.enabled = False
            logger.warning("Evidently not available")
            return
        
        self.enabled = True
        self.reference_data = reference_data
        logger.success("📊 Model Monitor (Evidently) initialized")
    
    def detect_data_drift(self, current_data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """
        Detecta drift nos dados
        """
        if not self.enabled or self.reference_data is None:
            return {'error': 'Not enabled or no reference data'}
        
        logger.info("🔍 Detecting data drift...")
        
        # Creatand report
        report = Report(metrics=[
            DataDriftPreset(),
        ])
        
        column_mapping = ColumnMapping()
        column_mapping.numerical_features = feature_columns
        
        # Run report
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Get results
        results = report.as_dict()
        
        # Extract key metrics
        drift_detected = False
        drifted_features = []
        
        if 'metrics' in results:
            for metric in results['metrics']:
                if 'result' in metric and 'drift_detected' in metric['result']:
                    if metric['result']['drift_detected']:
                        drift_detected = True
                        if 'column_name' in metric['result']:
                            drifted_features.append(metric['result']['column_name'])
        
        logger.info(f"Drift detected: {drift_detected}, Features: {len(drifted_features)}")
        
        return {
            'drift_detected': drift_detected,
            'drifted_features': drifted_features,
            'timestamp': datetime.now().isoformat(),
            'n_features': len(feature_columns),
            'n_drifted': len(drifted_features)
        }
    
    def monitor_model_performance(
        self,
        current_data: pd.DataFrame,
        prediction_column: str,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Monitora performance do modelo
        """
        if not self.enabled:
            return {'error': 'Not enabled'}
        
        logger.info("📈 Monitoring model performance...")
        
        report = Report(metrics=[
            TargetDriftPreset(),
            DataQualityPreset(),
        ])
        
        column_mapping = ColumnMapping()
        column_mapping.prediction = prediction_column
        column_mapping.target = target_column
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Savand report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/evidently_report_{timestamp}.html"
        report.save_html(report_path)
        
        logger.success(f"✅ Performance report saved: {report_path}")
        
        return {
            'report_path': report_path,
            'timestamp': datetime.now().isoformat()
        }


class AlertQualityMonitor:
    """
    Monitora qualidade dos alerts gerados
    """
    
    def __init__(self):
        self.alert_history = []
        logger.info("🎯 Alert Quality Monitor initialized")
    
    def track_alert(self, alert: Dict[str, Any], outcome: str):
        """
        Registra alert e outcome (true positive, false positive, etc)
        
        Outcomes: true_positive, false_positive, true_negative, false_negative
        """
        self.alert_history.append({
            'alert_id': alert.get('alert_id'),
            'created_at': alert.get('created_at'),
            'risk_level': alert.get('risk_level'),
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de qualidade dos alerts
        """
        if not self.alert_history:
            return {}
        
        tp = sum(1 for a in self.alert_history if a['outcome'] == 'true_positive')
        fp = sum(1 for a in self.alert_history if a['outcome'] == 'false_positive')
        tn = sum(1 for a in self.alert_history if a['outcome'] == 'true_negative')
        fn = sum(1 for a in self.alert_history if a['outcome'] == 'false_negative')
        
        total = tp + fp + tn + fn
        
        if total == 0:
            return {}
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_alerts': total
        }

