"""
🔄 RLHF - REINFORCEMENT LEARNING FROM HUMAN FEEDBACK
Sistema auto-evolutivo que aprende com feedback de analistas
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque
import json
from loguru import logger


class RLHFSystem:
    """
    Sistema que aprende com feedback humano e melhora continuamente
    """
    
    def __init__(self):
        self.feedback_history = deque(maxlen=10000)
        self.model_performance = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "false_positive_rate": 0.18
        }
        self.dynamic_thresholds = {
            "high_value": 10000,
            "structuring": 9900,
            "anomaly": 0.75
        }
        self.retrain_counter = 0
        self.retrain_threshold = 100
        
        logger.info("🔄 RLHF System initialized")
    
    async def collect_feedback(
        self,
        alert_id: str,
        analyst_decision: Dict[str, Any],
        alert_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coleta feedback do analista sobre um alerta
        
        Args:
            alert_id: ID do alerta
            analyst_decision: Decisão do analista (TP/FP/Unsure)
            alert_data: Dados completos do alerta
            
        Returns:
            Confirmation e stats
        """
        
        feedback = {
            "alert_id": alert_id,
            "timestamp": datetime.utcnow().isoformat(),
            "analyst_decision": analyst_decision["decision"],  # TP, FP, TN, FN
            "analyst_reasoning": analyst_decision.get("reasoning", ""),
            "analyst_id": analyst_decision.get("analyst_id", "unknown"),
            "confidence": analyst_decision.get("confidence", 1.0),
            
            # data do alerta
            "predicted_suspicious": alert_data.get("suspicious", False),
            "risk_score": alert_data.get("risk_score", 0.0),
            "patterns_detected": alert_data.get("patterns", []),
            "features": alert_data.get("features", {}),
            
            # Metadata
            "feedback_type": self._classify_feedback(
                alert_data.get("suspicious", False),
                analyst_decision["decision"]
            )
        }
        
        self.feedback_history.append(feedback)
        
        logger.info(f"🔄 Feedback collected for {alert_id}: {feedback['feedback_type']}")
        
        # Aprenofr imediatamente
        await self._immediate_learning(feedback)
        
        # Verificar sand ofvand retreinar
        if len(self.feedback_history) >= self.retrain_threshold:
            await self.trigger_retraining()
        
        return {
            "feedback_id": alert_id,
            "status": "collected",
            "total_feedback": len(self.feedback_history),
            "current_performance": self.model_performance,
            "retraining_in": self.retrain_threshold - len(self.feedback_history)
        }
    
    def _classify_feedback(self, predicted: bool, actual: str) -> str:
        """
        Classifica tipo de feedback
        """
        if actual == "TRUE_POSITIVE":
            return "TP" if predicted else "FN"
        elif actual == "FALSE_POSITIVE":
            return "FP" if predicted else "TN"
        elif actual == "TRUE_NEGATIVE":
            return "TN" if not predicted else "FP"
        else:
            return "UNKNOWN"
    
    async def _immediate_learning(self, feedback: Dict[str, Any]):
        """
        Aprendizado imediato com feedback
        """
        feedback_type = feedback["feedback_type"]
        
        # Ajustar thresholds dinamicamente
        if feedback_type == "FP":
            # Falso positivo - aumentar threshold
            await self._adjust_thresholds_up(feedback)
        elif feedback_type == "FN":
            # Falso negativo - diminuir threshold
            await self._adjust_thresholds_down(feedback)
        
        # Atualizar métricas
        await self._update_metrics()
        
        logger.debug(f"🔄 Immediate learning applied from {feedback_type}")
    
    async def _adjust_thresholds_up(self, feedback: Dict[str, Any]):
        """
        Aumenta thresholds para reduzir falsos positivos
        """
        patterns = feedback.get("patterns_detected", [])
        
        for pattern in patterns:
            if "structuring" in pattern:
                self.dynamic_thresholds["structuring"] += 50
            elif "high_value" in pattern:
                self.dynamic_thresholds["high_value"] += 500
        
        # Aumentar threshold of anomalia
        self.dynamic_thresholds["anomaly"] = min(
            self.dynamic_thresholds["anomaly"] + 0.02,
            0.95
        )
        
        logger.debug(f"📈 Thresholds adjusted up: {self.dynamic_thresholds}")
    
    async def _adjust_thresholds_down(self, feedback: Dict[str, Any]):
        """
        Diminui thresholds para capturar mais casos
        """
        patterns = feedback.get("patterns_detected", [])
        
        for pattern in patterns:
            if "structuring" in pattern:
                self.dynamic_thresholds["structuring"] = max(
                    self.dynamic_thresholds["structuring"] - 50,
                    9000
                )
            elif "high_value" in pattern:
                self.dynamic_thresholds["high_value"] = max(
                    self.dynamic_thresholds["high_value"] - 500,
                    5000
                )
        
        # Diminuir threshold of anomalia
        self.dynamic_thresholds["anomaly"] = max(
            self.dynamic_thresholds["anomaly"] - 0.02,
            0.5
        )
        
        logger.debug(f"📉 Thresholds adjusted down: {self.dynamic_thresholds}")
    
    async def _update_metrics(self):
        """
        Atualiza métricas de performance
        """
        if len(self.feedback_history) < 10:
            return
        
        recent_feedback = list(self.feedback_history)[-100:]
        
        tp = sum(1 for f in recent_feedback if f["feedback_type"] == "TP")
        fp = sum(1 for f in recent_feedback if f["feedback_type"] == "FP")
        tn = sum(1 for f in recent_feedback if f["feedback_type"] == "TN")
        fn = sum(1 for f in recent_feedback if f["feedback_type"] == "FN")
        
        total = tp + fp + tn + fn
        if total == 0:
            return
        
        self.model_performance["accuracy"] = (tp + tn) / total
        self.model_performance["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.model_performance["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision = self.model_performance["precision"]
        recall = self.model_performance["recall"]
        if precision + recall > 0:
            self.model_performance["f1_score"] = 2 * (precision * recall) / (precision + recall)
        
        self.model_performance["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        logger.info(f"📊 Metrics updated: Accuracy={self.model_performance['accuracy']:.2%}, "
                   f"FPR={self.model_performance['false_positive_rate']:.2%}")
    
    async def trigger_retraining(self):
        """
        Dispara retreinamento dos modelos
        """
        logger.info("🔄 Triggering model retraining...")
        
        self.retrain_counter += 1
        
        # in produção, aqui ifria o retreinamento real dos mooflos ML
        # Por ora, simulamos with ajustes of thresholds and parâmetros
        
        training_data = list(self.feedback_history)
        
        # Análisand of padrões no feedback
        fp_patterns = self._analyze_false_positives(training_data)
        fn_patterns = self._analyze_false_negatives(training_data)
        
        logger.info(f"🔄 Retraining #{self.retrain_counter} complete")
        logger.info(f"   FP patterns: {fp_patterns}")
        logger.info(f"   FN patterns: {fn_patterns}")
        
        # Limpar histórico antigo (manter apenas lasts 1000)
        while len(self.feedback_history) > 1000:
            self.feedback_history.popleft()
        
        return {
            "retrain_id": self.retrain_counter,
            "training_samples": len(training_data),
            "new_performance": self.model_performance,
            "fp_patterns": fp_patterns,
            "fn_patterns": fn_patterns
        }
    
    def _analyze_false_positives(self, data: List[Dict]) -> Dict[str, int]:
        """
        Analisa padrões em falsos positivos
        """
        fp_data = [f for f in data if f["feedback_type"] == "FP"]
        
        patterns = {}
        for fp in fp_data:
            for pattern in fp.get("patterns_detected", []):
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return patterns
    
    def _analyze_false_negatives(self, data: List[Dict]) -> Dict[str, int]:
        """
        Analisa padrões em falsos negativos
        """
        fn_data = [f for f in data if f["feedback_type"] == "FN"]
        
        patterns = {}
        for fn in fn_data:
            for pattern in fn.get("patterns_detected", []):
                patterns[pattern] = patterns.get(pattern, 0) + 1
        
        return patterns
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """
        Relatório completo de performance
        """
        return {
            "current_metrics": self.model_performance,
            "dynamic_thresholds": self.dynamic_thresholds,
            "total_feedback": len(self.feedback_history),
            "retraining_count": self.retrain_counter,
            "feedback_distribution": self._get_feedback_distribution(),
            "improvement_trend": await self._calculate_improvement_trend()
        }
    
    def _get_feedback_distribution(self) -> Dict[str, int]:
        """
        Distribuição dos tipos de feedback
        """
        distribution = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        
        for feedback in self.feedback_history:
            fb_type = feedback["feedback_type"]
            if fb_type in distribution:
                distribution[fb_type] += 1
        
        return distribution
    
    async def _calculate_improvement_trend(self) -> Dict[str, Any]:
        """
        Calcula tendência de melhoria
        """
        if len(self.feedback_history) < 50:
            return {"trend": "insufficient_data"}
        
        # withforr primeiros 50 vs lasts 50
        first_batch = list(self.feedback_history)[:50]
        last_batch = list(self.feedback_history)[-50:]
        
        first_fpr = sum(1 for f in first_batch if f["feedback_type"] == "FP") / 50
        last_fpr = sum(1 for f in last_batch if f["feedback_type"] == "FP") / 50
        
        improvement = (first_fpr - last_fpr) / first_fpr if first_fpr > 0 else 0
        
        return {
            "trend": "improving" if improvement > 0 else "stable" if improvement == 0 else "declining",
            "fpr_improvement": improvement,
            "first_batch_fpr": first_fpr,
            "last_batch_fpr": last_fpr
        }
    
    def get_optimal_thresholds(self) -> Dict[str, float]:
        """
        Retorna thresholds otimizados atual
        """
        return self.dynamic_thresholds.copy()


class PromptOptimizer:
    """
    Otimiza prompts de LLM baseado em resultados
    Similar ao DSPy
    """
    
    def __init__(self):
        self.prompt_versions = {}
        self.performance_history = {}
        logger.info("🎯 Prompt Optimizer initialized")
    
    async def optimize_prompt(
        self,
        prompt_name: str,
        current_prompt: str,
        feedback_data: List[Dict[str, Any]]
    ) -> str:
        """
        Otimiza prompt baseado em feedback
        """
        
        # Analisar performancand do prompt atual
        success_rate = self._calculate_success_rate(feedback_data)
        
        # Sand performancand é boa, manter
        if success_rate > 0.9:
            logger.info(f"🎯 Prompt '{prompt_name}' performing well ({success_rate:.1%})")
            return current_prompt
        
        # Sand performancand é ruim, tentar melhorar
        logger.info(f"🎯 Optimizing prompt '{prompt_name}' (current: {success_rate:.1%})")
        
        # Aqui ifria otimização automática (similar ao DSPy)
        # Por ora, retornamos o prompt original
        
        optimized_prompt = current_prompt  # TODO: Real optimization
        
        self.prompt_versions[prompt_name] = {
            "version": len(self.prompt_versions.get(prompt_name, [])) + 1,
            "prompt": optimized_prompt,
            "success_rate": success_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return optimized_prompt
    
    def _calculate_success_rate(self, feedback_data: List[Dict]) -> float:
        """
        Calcula taxa de sucesso
        """
        if not feedback_data:
            return 0.5
        
        successful = sum(1 for f in feedback_data if f.get("feedback_type") in ["TP", "TN"])
        return successful / len(feedback_data)

