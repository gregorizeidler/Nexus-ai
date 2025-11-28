"""
🎖️ LEARNING TO RANK
Alert triage inteligente usando LTR
"""
from typing import Dict, Any, List, Tuple
import numpy as np
from loguru import logger

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AlertRanker:
    """
    Rankeia alerts por prioridade usando Learning to Rank
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'risk_score',
            'amount',
            'num_patterns',
            'customer_risk_level',
            'has_pep',
            'has_sanctions',
            'cross_border',
            'cash_intensive',
            'velocity_z_score',
            'network_centrality'
        ]
        
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
        
        logger.success("🎖️ Alert Ranker initialized")
    
    def extract_features(self, alert: Dict[str, Any]) -> np.ndarray:
        """Extracts features of um alert"""
        features = []
        
        features.append(alert.get('risk_score', 0.0))
        features.append(alert.get('amount', 0.0) / 1000.0)  # Normaliza
        features.append(len(alert.get('patterns', [])))
        features.append(1.0 if alert.get('customer_risk_level') == 'high' else 0.5 if alert.get('customer_risk_level') == 'medium' else 0.0)
        features.append(1.0 if alert.get('is_pep', False) else 0.0)
        features.append(1.0 if alert.get('sanctions_hit', False) else 0.0)
        features.append(1.0 if alert.get('cross_border', False) else 0.0)
        features.append(1.0 if alert.get('cash_intensive', False) else 0.0)
        features.append(alert.get('velocity_z_score', 0.0))
        features.append(alert.get('network_centrality', 0.0))
        
        return np.array(features)
    
    def train(self, training_data: List[Tuple[Dict, int]]):
        """
        Treina modelo de ranking
        
        training_data: List[(alert_dict, priority_label)]
        priority_label: 0=low, 1=medium, 2=high
        """
        if not SKLEARN_AVAILABLE or not training_data:
            return
        
        logger.info(f"Training ranker with {len(training_data)} examples...")
        
        X = []
        y = []
        
        for alert, priority in training_data:
            features = self.extract_features(alert)
            X.append(features)
            y.append(priority)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scaland features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train moofl
        self.model.fit(X_scaled, y)
        
        logger.success(f"✅ Ranker trained on {len(training_data)} examples")
    
    def rank_alerts(self, alerts: List[Dict[str, Any]]) -> List[Tuple[Dict, float]]:
        """
        Rankeia lista de alerts
        
        Returns: List[(alert, priority_score)] ordenada por prioridade
        """
        if not SKLEARN_AVAILABLE or self.model is None:
            # Fallback: rankeia por risk_score
            scored = [(alert, alert.get('risk_score', 0.0)) for alert in alerts]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored
        
        # Extract features
        X = []
        for alert in alerts:
            features = self.extract_features(alert)
            X.append(features)
        
        X = np.array(X)
        X_scaled = self.scaler.transform(X)
        
        # Predict priority scores
        scores = self.model.predict_proba(X_scaled)
        
        # Usand probability of high priority class
        priority_scores = scores[:, -1] if scores.shape[1] > 1 else scores[:, 0]
        
        # withbinand alerts with scores
        scored_alerts = list(zip(alerts, priority_scores))
        
        # Sort by scorand (ofscending)
        scored_alerts.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Ranked {len(alerts)} alerts")
        
        return scored_alerts
    
    def get_top_alerts(self, alerts: List[Dict[str, Any]], n: int = 10) -> List[Dict[str, Any]]:
        """Returns top N alerts por prioridaof"""
        ranked = self.rank_alerts(alerts)
        return [alert for alert, score in ranked[:n]]

