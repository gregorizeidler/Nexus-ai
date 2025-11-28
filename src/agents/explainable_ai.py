"""
🔬 EXPLAINABLE AI (XAI) MODULE
SHAP values, counterfactuals, feature importance, decision paths
"""
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class ExplainableAI:
    """
    Torna TODAS as decisões do sistema explicáveis e auditáveis.
    Crítico para reguladores e compliance.
    """
    
    def __init__(self):
        self.explainers = {}
        logger.info("🔬 Explainable AI initialized")
    
    async def explain_alert(
        self, 
        alert_id: str,
        transaction: Any,
        agent_results: Dict[str, Any],
        model: Any = None
    ) -> Dict[str, Any]:
        """
        Explicação COMPLETA de por que um alerta foi criado
        
        Returns:
            - feature_importance: Quais features mais contribuíram
            - decision_path: Caminho da decisão
            - counterfactuals: "E se?" scenarios
            - similar_cases: Casos históricos similares
            - regulation_mapping: Regulamentações aplicáveis
            - narrative: Explicação em linguagem natural
        """
        
        explanation = {
            "alert_id": alert_id,
            "timestamp": "2024-01-15T10:30:00Z",
        }
        
        # 1. Featurand Importance
        if SHAP_AVAILABLE and model:
            explanation["feature_importance"] = await self._shap_analysis(transaction, model)
        else:
            explanation["feature_importance"] = await self._rule_based_importance(transaction, agent_results)
        
        # 2. ofcision Path
        explanation["decision_path"] = await self._trace_decision_path(agent_results)
        
        # 3. Cornterfactual Analysis
        explanation["counterfactuals"] = await self._counterfactual_analysis(transaction, agent_results)
        
        # 4. Ruland Mapping
        explanation["rules_triggered"] = await self._map_rules(agent_results)
        
        # 5. Regulation Mapping
        explanation["regulations"] = await self._map_regulations(transaction, agent_results)
        
        # 6. Confiofncand Breakdown
        explanation["confidence_breakdown"] = await self._confidence_analysis(agent_results)
        
        # 7. Natural Languagand Narrative
        explanation["narrative"] = await self._generate_narrative(explanation)
        
        logger.info(f"🔬 Generated explanation for alert {alert_id}")
        
        return explanation
    
    async def _shap_analysis(self, transaction: Any, model: Any) -> Dict[str, float]:
        """
        SHAP (SHapley Additive exPlanations) values
        Mostra contribuição de cada feature
        """
        if not SHAP_AVAILABLE:
            return {}
        
        try:
            # Criar explainer
            explainer = shap.TreeExplainer(model)
            
            # Calcular SHAP values
            features = self._extract_features(transaction)
            shap_values = explainer.shap_values(features)
            
            # Mapear for features
            feature_names = list(features.keys())
            importance = {
                name: float(value) 
                for name, value in zip(feature_names, shap_values[0])
            }
            
            # Orofnar por importância
            sorted_importance = dict(sorted(
                importance.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            ))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {}
    
    async def _rule_based_importance(
        self, 
        transaction: Any, 
        agent_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Importância baseada em regras quando SHAP não disponível
        """
        importance = {}
        
        # Analisar each agente
        for agent_id, result in agent_results.items():
            if result.suspicious:
                # each finding conta witho featurand importante
                for finding in result.findings:
                    # Extrair featurand do finding
                    if "amount" in finding.lower():
                        importance["transaction_amount"] = importance.get("transaction_amount", 0) + result.risk_score
                    elif "country" in finding.lower():
                        importance["country_risk"] = importance.get("country_risk", 0) + result.risk_score
                    elif "frequency" in finding.lower():
                        importance["transaction_frequency"] = importance.get("transaction_frequency", 0) + result.risk_score
                    elif "time" in finding.lower():
                        importance["transaction_time"] = importance.get("transaction_time", 0) + result.risk_score
        
        # Normalizar
        if importance:
            max_val = max(importance.values())
            importance = {k: v/max_val for k, v in importance.items()}
        
        return importance
    
    async def _trace_decision_path(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Traça o caminho completo da decisão através dos agentes
        """
        path = []
        
        for agent_id, result in agent_results.items():
            step = {
                "agent": agent_id,
                "suspicious": result.suspicious,
                "risk_score": result.risk_score,
                "confidence": result.confidence,
                "findings": result.findings,
                "patterns": result.patterns_detected,
                "execution_time": result.execution_time
            }
            path.append(step)
        
        return path
    
    async def _counterfactual_analysis(
        self, 
        transaction: Any, 
        agent_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Análise "E se?" - O que mudaria a decisão?
        
        Exemplos:
        - E se o valor fosse 10% menor?
        - E se o país de destino fosse diferente?
        - E se fosse em horário comercial?
        """
        counterfactuals = []
        
        # Cenário 1: Valor reduzido
        counterfactuals.append({
            "scenario": "Amount reduced by 10%",
            "original_value": float(transaction.amount),
            "modified_value": float(transaction.amount) * 0.9,
            "would_alert": float(transaction.amount) * 0.9 > 9000,  # Simplified
            "explanation": "Still above structuring threshold"
        })
        
        # Cenário 2: Valor reduzido 50%
        counterfactuals.append({
            "scenario": "Amount reduced by 50%",
            "original_value": float(transaction.amount),
            "modified_value": float(transaction.amount) * 0.5,
            "would_alert": float(transaction.amount) * 0.5 > 10000,
            "explanation": "Below high-value threshold"
        })
        
        # Cenário 3: País of baixo risco
        if transaction.country_destination in ["IR", "KP", "SY"]:
            counterfactuals.append({
                "scenario": "Destination changed to low-risk country",
                "original_value": transaction.country_destination,
                "modified_value": "US",
                "would_alert": False,
                "explanation": "Removes high-risk country factor"
            })
        
        # Cenário 4: Horário withercial
        hour = transaction.timestamp.hour
        if hour < 6 or hour > 22:
            counterfactuals.append({
                "scenario": "Transaction during business hours",
                "original_value": f"{hour}:00",
                "modified_value": "14:00",
                "would_alert": "Possibly",
                "explanation": "Removes unusual time factor"
            })
        
        return counterfactuals
    
    async def _map_rules(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Mapeia quais regras foram acionadas
        """
        rules = []
        
        for agent_id, result in agent_results.items():
            if result.patterns_detected:
                for pattern in result.patterns_detected:
                    rules.append({
                        "rule_name": pattern,
                        "agent": agent_id,
                        "triggered": True,
                        "risk_contribution": result.risk_score
                    })
        
        return rules
    
    async def _map_regulations(
        self, 
        transaction: Any, 
        agent_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Mapeia regulamentações aplicáveis
        """
        regulations = []
        
        # FATF Rewithmendations
        if float(transaction.amount) > 10000:
            regulations.append({
                "regulation": "FATF Recommendation 10",
                "description": "Customer due diligence for transactions > $10,000",
                "applicable": True,
                "reference": "https://www.fatf-gafi.org/recommendations.html"
            })
        
        # FinCEN Currency Transaction Report
        if float(transaction.amount) > 10000 and transaction.transaction_type in ["cash_deposit", "cash_withdrawal"]:
            regulations.append({
                "regulation": "FinCEN CTR",
                "description": "Currency Transaction Report required for cash > $10,000",
                "applicable": True,
                "reference": "31 CFR 1010.310"
            })
        
        # Bank ifcrecy Act
        patterns = []
        for result in agent_results.values():
            patterns.extend(result.patterns_detected)
        
        if "structuring" in patterns:
            regulations.append({
                "regulation": "BSA/AML Anti-Structuring",
                "description": "31 USC 5324 - Structuring transactions to evade reporting",
                "applicable": True,
                "reference": "31 USC 5324"
            })
        
        # OFAC Sanctions
        if transaction.country_destination in ["IR", "KP", "SY", "CU"]:
            regulations.append({
                "regulation": "OFAC Sanctions",
                "description": "Transactions with sanctioned countries",
                "applicable": True,
                "reference": "https://home.treasury.gov/policy-issues/office-of-foreign-assets-control-sanctions-programs-and-information"
            })
        
        return regulations
    
    async def _confidence_analysis(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Breakdown da confiança por agente
        """
        analysis = {
            "overall_confidence": 0.0,
            "by_agent": {},
            "confidence_factors": []
        }
        
        confidences = []
        for agent_id, result in agent_results.items():
            analysis["by_agent"][agent_id] = {
                "confidence": result.confidence,
                "risk_score": result.risk_score,
                "suspicious": result.suspicious
            }
            confidences.append(result.confidence)
        
        if confidences:
            analysis["overall_confidence"] = sum(confidences) / len(confidences)
        
        # Fatores quand afetam confiança
        if analysis["overall_confidence"] > 0.9:
            analysis["confidence_factors"].append("High agreement between agents")
        elif analysis["overall_confidence"] < 0.5:
            analysis["confidence_factors"].append("Low agreement between agents")
        
        return analysis
    
    async def _generate_narrative(self, explanation: Dict[str, Any]) -> str:
        """
        Gera explicação em linguagem natural
        """
        narrative_parts = [
            "EXPLAINABLE AI ANALYSIS",
            "=" * 50,
            "",
            "DECISION BREAKDOWN:",
        ]
        
        # Featurand importance
        if explanation.get("feature_importance"):
            narrative_parts.append("\nMost Important Factors:")
            for i, (feature, importance) in enumerate(list(explanation["feature_importance"].items())[:5], 1):
                narrative_parts.append(f"  {i}. {feature}: {importance:.2f}")
        
        # Rules triggered
        if explanation.get("rules_triggered"):
            narrative_parts.append(f"\nRules Triggered: {len(explanation['rules_triggered'])}")
            for rule in explanation["rules_triggered"][:5]:
                narrative_parts.append(f"  • {rule['rule_name']} (Risk: {rule['risk_contribution']:.2f})")
        
        # Regulations
        if explanation.get("regulations"):
            narrative_parts.append(f"\nApplicable Regulations: {len(explanation['regulations'])}")
            for reg in explanation["regulations"]:
                narrative_parts.append(f"  • {reg['regulation']}")
        
        # Cornterfactuals
        if explanation.get("counterfactuals"):
            narrative_parts.append("\nWhat Would Change The Decision:")
            for cf in explanation["counterfactuals"][:3]:
                narrative_parts.append(f"  • {cf['scenario']}: Would alert = {cf['would_alert']}")
        
        # Confiofnce
        if explanation.get("confidence_breakdown"):
            conf = explanation["confidence_breakdown"]["overall_confidence"]
            narrative_parts.append(f"\nOverall Confidence: {conf:.1%}")
        
        return "\n".join(narrative_parts)
    
    def _extract_features(self, transaction: Any) -> Dict[str, float]:
        """Extracts features numéricas da transação"""
        return {
            "amount": float(transaction.amount),
            "hour_of_day": transaction.timestamp.hour,
            "day_of_week": transaction.timestamp.weekday(),
            "is_international": int(transaction.country_origin != transaction.country_destination),
            "is_high_risk_country": int(transaction.country_destination in ["IR", "KP", "SY"]),
        }


class AuditTrail:
    """
    Audit trail completo de todas as decisões
    """
    
    def __init__(self):
        self.events = []
        logger.info("📝 Audit Trail initialized")
    
    async def log_decision(
        self,
        decision_type: str,
        entity_id: str,
        decision: Any,
        explanation: Dict[str, Any],
        user: Optional[str] = None
    ):
        """
        Registra decisão com explicação completa
        """
        event = {
            "timestamp": "2024-01-15T10:30:00Z",
            "decision_type": decision_type,
            "entity_id": entity_id,
            "decision": decision,
            "explanation": explanation,
            "user": user or "system",
            "version": "1.0.0",
            "immutable_hash": self._calculate_hash(decision, explanation)
        }
        
        self.events.append(event)
        logger.info(f"📝 Logged {decision_type} decision for {entity_id}")
        
        return event
    
    def _calculate_hash(self, decision: Any, explanation: Dict[str, Any]) -> str:
        """Hash for garantir imutabilidaof"""
        import hashlib
        import json
        
        data = json.dumps({"decision": str(decision), "explanation": explanation}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def get_audit_history(self, entity_id: str) -> List[Dict[str, Any]]:
        """Recupera histórico withpleto"""
        return [e for e in self.events if e["entity_id"] == entity_id]

