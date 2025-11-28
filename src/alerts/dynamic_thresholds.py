"""
🎯 DYNAMIC THRESHOLDS
Adaptive learning para thresholds de detecção
"""
from typing import Dict, Any, List
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from loguru import logger


class AdaptiveThresholdManager:
    """
    Gerencia thresholds adaptativos baseado em feedback
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.thresholds = {}
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.feedback_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0})
        
        # Initial thresholds
        self.thresholds = {
            'large_amount': 10000.0,
            'frequent_transactions': 10,
            'structuring_threshold': 9500.0,
            'velocity_multiplier': 3.0,
            'risk_score': 0.7
        }
        
        logger.success("🎯 Adaptive Threshold Manager initialized")
    
    def get_threshold(self, threshold_name: str) -> float:
        """Returns threshold atual"""
        return self.thresholds.get(threshold_name, 0.0)
    
    def record_detection(self, threshold_name: str, value: float, was_triggered: bool):
        """Registra uma oftecção"""
        self.history[threshold_name].append({
            'value': value,
            'triggered': was_triggered,
            'timestamp': datetime.now()
        })
    
    def record_feedback(self, threshold_name: str, feedback_type: str):
        """
        Registra feedback sobre uma detecção
        
        feedback_type: 'tp', 'fp', 'tn', 'fn'
        """
        self.feedback_counts[threshold_name][feedback_type] += 1
        
        # Adjust threshold baifd on feedback
        self._adjust_threshold(threshold_name)
    
    def _adjust_threshold(self, threshold_name: str):
        """
        Ajusta threshold baseado em feedback
        """
        counts = self.feedback_counts[threshold_name]
        
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        
        total = tp + fp + fn
        if total < 10:  # Espera mínimo de feedback
            return
        
        # Calculates precisão and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        current_threshold = self.thresholds[threshold_name]
        
        # Ajustand adaptativo
        if precision < 0.5:  # Muitos falsos positivos
            # Aumenta threshold for reduzir FP
            new_threshold = current_threshold * 1.1
            logger.info(f"📈 Increasing {threshold_name}: {current_threshold:.2f} -> {new_threshold:.2f} (low precision)")
            
        elif recall < 0.5:  # Muitos falsos negativos
            # Diminui threshold for capturar mais
            new_threshold = current_threshold * 0.9
            logger.info(f"📉 Decreasing {threshold_name}: {current_threshold:.2f} -> {new_threshold:.2f} (low recall)")
            
        else:
            # Threshold está bom
            new_threshold = current_threshold
        
        self.thresholds[threshold_name] = new_threshold
    
    def get_statistics(self, threshold_name: str) -> Dict[str, Any]:
        """Estatísticas of um threshold"""
        history = list(self.history[threshold_name])
        
        if not history:
            return {}
        
        values = [h['value'] for h in history]
        triggered = [h['triggered'] for h in history]
        
        counts = self.feedback_counts[threshold_name]
        tp, fp, tn, fn = counts['tp'], counts['fp'], counts['tn'], counts['fn']
        total = tp + fp + tn + fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'current_threshold': self.thresholds[threshold_name],
            'n_observations': len(values),
            'mean_value': np.mean(values),
            'std_value': np.std(values),
            'trigger_rate': sum(triggered) / len(triggered),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_feedback': total
        }


class BehavioralBaseline:
    """
    Calcula baseline comportamental por cliente
    """
    
    def __init__(self, learning_period_days: int = 30):
        self.learning_period_days = learning_period_days
        self.customer_baselines = {}
        logger.info(f"📊 Behavioral Baseline initialized (learning period: {learning_period_days} days)")
    
    def update_baseline(self, customer_id: str, transaction_data: Dict[str, Any]):
        """Updates baiflinand of um cliente"""
        if customer_id not in self.customer_baselines:
            self.customer_baselines[customer_id] = {
                'transactions': deque(maxlen=1000),
                'amount_mean': 0.0,
                'amount_std': 0.0,
                'frequency': 0.0,
                'last_updated': datetime.now()
            }
        
        baseline = self.customer_baselines[customer_id]
        baseline['transactions'].append(transaction_data)
        
        # Recalcula estatísticas
        amounts = [t['amount'] for t in baseline['transactions']]
        baseline['amount_mean'] = np.mean(amounts)
        baseline['amount_std'] = np.std(amounts)
        
        # Frethatncy (txn per day)
        if len(baseline['transactions']) > 1:
            first_txn = min(t['timestamp'] for t in baseline['transactions'])
            last_txn = max(t['timestamp'] for t in baseline['transactions'])
            days = (last_txn - first_txn).days or 1
            baseline['frequency'] = len(baseline['transactions']) / days
        
        baseline['last_updated'] = datetime.now()
    
    def get_deviation(self, customer_id: str, amount: float) -> float:
        """
        Calcula desvio do comportamento normal
        
        Returns: Z-score (número de desvios padrão)
        """
        if customer_id not in self.customer_baselines:
            return 0.0
        
        baseline = self.customer_baselines[customer_id]
        
        if baseline['amount_std'] == 0:
            return 0.0
        
        z_score = (amount - baseline['amount_mean']) / baseline['amount_std']
        return abs(z_score)
    
    def is_anomalous(self, customer_id: str, amount: float, threshold_stds: float = 3.0) -> bool:
        """
        Check se transação é anômala
        
        threshold_stds: número de desvios padrão para considerar anômalo
        """
        deviation = self.get_deviation(customer_id, amount)
        return deviation > threshold_stds

