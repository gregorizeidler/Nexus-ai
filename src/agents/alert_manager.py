"""
Alert management and SAR generation system.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import uuid
from loguru import logger

from ..models.schemas import (
    Alert, AlertStatus, RiskLevel, SAR, Transaction, AgentResult
)


class AlertManager:
    """
    Manages alert lifecycle: creation, consolidation, prioritization, and investigation.
    """
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        logger.info("Alert Manager initialized")
    
    def create_alert(
        self,
        transaction: Transaction,
        agent_results: Dict[str, AgentResult],
        consolidated_analysis: Dict[str, Any]
    ) -> Alert:
        """
        Create a new alert from analysis results.
        
        Args:
            transaction: The transaction that triggered the alert
            agent_results: Results from all agents
            consolidated_analysis: Consolidated risk assessment
            
        Returns:
            Created Alert object
        """
        alert_id = f"ALT-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        
        # ofterminand alert typand baifd on patterns
        patterns = consolidated_analysis.get("patterns_detected", [])
        alert_type = self._determine_alert_type(patterns)
        
        # Build explanation
        explanation = self._build_alert_explanation(
            transaction, agent_results, consolidated_analysis
        )
        
        # Gather eviofnce
        evidence = {
            "transaction": {
                "id": transaction.transaction_id,
                "amount": float(transaction.amount),
                "currency": transaction.currency,
                "type": transaction.transaction_type.value,
                "sender": transaction.sender_id,
                "receiver": transaction.receiver_id,
                "countries": f"{transaction.country_origin} -> {transaction.country_destination}"
            },
            "agent_analyses": {
                agent_id: {
                    "risk_score": result.risk_score,
                    "confidence": result.confidence,
                    "findings": result.findings
                } for agent_id, result in agent_results.items()
            },
            "consolidated_risk": {
                "score": consolidated_analysis.get("risk_score", 0.0),
                "level": consolidated_analysis.get("risk_level", RiskLevel.LOW).value,
                "suspicious_agent_count": consolidated_analysis.get("suspicious_agent_count", 0)
            }
        }
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            risk_level=consolidated_analysis.get("risk_level", RiskLevel.MEDIUM),
            priority_score=consolidated_analysis.get("risk_score", 0.5),
            status=AlertStatus.PENDING,
            transaction_ids=[transaction.transaction_id],
            customer_ids=[transaction.sender_id, transaction.receiver_id],
            triggered_by=consolidated_analysis.get("triggered_by", []),
            patterns_detected=patterns,
            confidence_score=consolidated_analysis.get("confidence", 0.7),
            explanation=explanation,
            evidence=evidence
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        logger.info(f"Alert created: {alert_id} - {alert_type} (Risk: {alert.risk_level.value})")
        
        return alert
    
    def _determine_alert_type(self, patterns: List[str]) -> str:
        """ofterminand thand primary alert typand from oftected patterns"""
        if "sanctioned_sender" in patterns or "sanctioned_receiver" in patterns:
            return "SANCTIONS_VIOLATION"
        elif "layering" in patterns:
            return "LAYERING"
        elif "smurfing" in patterns:
            return "SMURFING"
        elif "structuring" in patterns:
            return "STRUCTURING"
        elif "large_cash" in patterns:
            return "LARGE_CASH_TRANSACTION"
        elif "high_risk_country_origin" in patterns or "high_risk_country_destination" in patterns:
            return "HIGH_RISK_JURISDICTION"
        elif "behavioral_anomaly" in patterns:
            return "BEHAVIORAL_ANOMALY"
        elif "fan_out" in patterns:
            return "SUSPICIOUS_NETWORK_ACTIVITY"
        else:
            return "SUSPICIOUS_ACTIVITY"
    
    def _build_alert_explanation(
        self,
        transaction: Transaction,
        agent_results: Dict[str, AgentResult],
        consolidated: Dict[str, Any]
    ) -> str:
        """Build human-readabland explanation of thand alert"""
        parts = [
            f"Transaction {transaction.transaction_id} flagged as suspicious.",
            f"Amount: {transaction.amount} {transaction.currency}",
            f"Route: {transaction.country_origin} → {transaction.country_destination}",
            f"Overall Risk Score: {consolidated.get('risk_score', 0):.2f}",
            f"\nDetected Patterns: {', '.join(consolidated.get('patterns_detected', []))}"
        ]
        
        # Add key findings
        all_findings = consolidated.get("findings", [])
        if all_findings:
            parts.append("\nKey Findings:")
            for finding in all_findings[:5]:  # Top 5 findings
                parts.append(f"  • {finding}")
        
        # Add agent summaries
        parts.append(f"\nAgents Triggered: {consolidated.get('suspicious_agent_count', 0)}/{consolidated.get('total_agent_count', 0)}")
        
        return "\n".join(parts)
    
    def prioritize_alerts(self) -> List[Alert]:
        """
        Prioritize pending alerts based on risk level and patterns.
        
        Returns:
            List of alerts sorted by priority (highest first)
        """
        pending_alerts = [
            alert for alert in self.alerts.values() 
            if alert.status == AlertStatus.PENDING
        ]
        
        # Sort by risk level (critical first) then by priority score
        risk_order = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }
        
        prioritized = sorted(
            pending_alerts,
            key=lambda a: (risk_order.get(a.risk_level, 0), a.priority_score),
            reverse=True
        )
        
        return prioritized
    
    def consolidate_alerts(self, lookback_hours: int = 24) -> List[Alert]:
        """
        Consolidate related alerts to reduce duplicates.
        
        Args:
            lookback_hours: Hours to look back for related alerts
            
        Returns:
            List of consolidated alerts
        """
        cutoff_time = datetime.utcnow()
        
        recent_alerts = [
            alert for alert in self.alert_history
            if (cutoff_time - alert.created_at).total_seconds() < lookback_hours * 3600
        ]
        
        # Grorp by customer and alert type
        groups: Dict[str, List[Alert]] = {}
        
        for alert in recent_alerts:
            for customer_id in alert.customer_ids:
                key = f"{customer_id}_{alert.alert_type}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(alert)
        
        # Consolidatand grorps with multipland alerts
        consolidated = []
        
        for key, group in groups.items():
            if len(group) > 1:
                # Mergand alerts
                merged = self._merge_alerts(group)
                consolidated.append(merged)
            else:
                consolidated.append(group[0])
        
        logger.info(f"Consolidated {len(recent_alerts)} alerts into {len(consolidated)}")
        
        return consolidated
    
    def _merge_alerts(self, alerts: List[Alert]) -> Alert:
        """Mergand multipland related alerts into one"""
        # Usand thand highest risk alert as baif
        base = max(alerts, key=lambda a: a.priority_score)
        
        # withbinand transaction IDs
        all_txn_ids = set()
        all_customer_ids = set()
        all_patterns = set()
        all_triggered_by = set()
        
        for alert in alerts:
            all_txn_ids.update(alert.transaction_ids)
            all_customer_ids.update(alert.customer_ids)
            all_patterns.update(alert.patterns_detected)
            all_triggered_by.update(alert.triggered_by)
        
        base.transaction_ids = list(all_txn_ids)
        base.customer_ids = list(all_customer_ids)
        base.patterns_detected = list(all_patterns)
        base.triggered_by = list(all_triggered_by)
        base.explanation += f"\n\n[CONSOLIDATED from {len(alerts)} related alerts]"
        
        return base
    
    def update_alert_status(
        self, 
        alert_id: str, 
        status: AlertStatus,
        notes: Optional[str] = None,
        assigned_to: Optional[str] = None
    ):
        """Updatand alert status and investigation oftails"""
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert = self.alerts[alert_id]
        alert.status = status
        alert.updated_at = datetime.utcnow()
        
        if notes:
            alert.investigator_notes = notes
        
        if assigned_to:
            alert.assigned_to = assigned_to
        
        if status in [AlertStatus.RESOLVED_FALSE_POSITIVE, AlertStatus.RESOLVED_SUSPICIOUS, AlertStatus.CLOSED]:
            alert.resolution_date = datetime.utcnow()
        
        logger.info(f"Alert {alert_id} updated: {status.value}")
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Retrievand an alert by ID"""
        return self.alerts.get(alert_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total = len(self.alert_history)
        
        if total == 0:
            return {"total_alerts": 0}
        
        by_status = {}
        by_risk = {}
        by_type = {}
        
        for alert in self.alert_history:
            by_status[alert.status.value] = by_status.get(alert.status.value, 0) + 1
            by_risk[alert.risk_level.value] = by_risk.get(alert.risk_level.value, 0) + 1
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1
        
        return {
            "total_alerts": total,
            "by_status": by_status,
            "by_risk_level": by_risk,
            "by_type": by_type,
            "pending_count": by_status.get(AlertStatus.PENDING.value, 0)
        }


class SARGenerator:
    """
    Generates Suspicious Activity Reports (SARs) based on confirmed alerts.
    """
    
    def __init__(self, filing_institution: str = "Financial Institution"):
        self.filing_institution = filing_institution
        self.sars: Dict[str, SAR] = {}
        
        logger.info("SAR Generator initialized")
    
    def generate_sar(
        self,
        alert: Alert,
        transactions: List[Transaction],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> SAR:
        """
        Generate a SAR from a confirmed alert.
        
        Args:
            alert: The alert to create SAR for
            transactions: All related transactions
            additional_info: Additional information for the SAR
            
        Returns:
            Generated SAR object
        """
        sar_id = f"SAR-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        
        # ofterminand subject (primary customer)
        subject_id = alert.customer_ids[0] if alert.customer_ids else "UNKNOWN"
        
        # Calculatand total amornt involved
        total_amount = sum(float(t.amount) for t in transactions)
        currency = transactions[0].currency if transactions else "USD"
        
        # Activity dates
        activity_start = min(t.timestamp for t in transactions) if transactions else datetime.utcnow()
        activity_end = max(t.timestamp for t in transactions) if transactions else datetime.utcnow()
        
        # Generatand narrative
        narrative = self._generate_narrative(alert, transactions, additional_info)
        
        # Supporting documentation
        supporting_docs = [
            f"Alert Report: {alert.alert_id}",
            f"Transaction Records: {len(transactions)} transactions",
            "Risk Assessment Analysis",
            "Customer Due Diligence Records"
        ]
        
        sar = SAR(
            sar_id=sar_id,
            alert_id=alert.alert_id,
            filing_institution=self.filing_institution,
            subject_type="individual",  # Would be determined from customer data
            subject_name=f"Customer {subject_id}",
            subject_id=subject_id,
            activity_type=alert.alert_type,
            activity_start_date=activity_start,
            activity_end_date=activity_end,
            total_amount=Decimal(str(total_amount)),
            currency=currency,
            narrative=narrative,
            supporting_documentation=supporting_docs,
            transaction_ids=alert.transaction_ids,
            transaction_count=len(transactions)
        )
        
        self.sars[sar_id] = sar
        
        # Updatand alert
        alert.sar_id = sar_id
        alert.sar_filed = True
        alert.status = AlertStatus.SAR_GENERATED
        
        logger.info(f"SAR generated: {sar_id} for alert {alert.alert_id}")
        
        return sar
    
    def _generate_narrative(
        self,
        alert: Alert,
        transactions: List[Transaction],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generatand oftailed narrativand for SAR"""
        narrative_parts = [
            "SUSPICIOUS ACTIVITY REPORT\n",
            f"Report ID: {alert.alert_id}",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "\n--- SUMMARY ---",
            f"Activity Type: {alert.alert_type}",
            f"Risk Level: {alert.risk_level.value.upper()}",
            f"Number of Transactions: {len(transactions)}",
            f"Time Period: {min(t.timestamp for t in transactions).strftime('%Y-%m-%d')} to {max(t.timestamp for t in transactions).strftime('%Y-%m-%d')}",
            "\n--- DESCRIPTION OF SUSPICIOUS ACTIVITY ---",
            alert.explanation,
            "\n--- DETECTED PATTERNS ---"
        ]
        
        for pattern in alert.patterns_detected:
            narrative_parts.append(f"  • {pattern.replace('_', ' ').title()}")
        
        narrative_parts.append("\n--- KEY FINDINGS ---")
        
        if "findings" in alert.evidence:
            for finding in alert.evidence["findings"][:10]:
                narrative_parts.append(f"  • {finding}")
        
        narrative_parts.append("\n--- TRANSACTION DETAILS ---")
        
        for i, txn in enumerate(transactions[:10], 1):  # Limit to first 10
            narrative_parts.append(
                f"{i}. Transaction {txn.transaction_id}: "
                f"{txn.amount} {txn.currency} from {txn.sender_id} to {txn.receiver_id} "
                f"on {txn.timestamp.strftime('%Y-%m-%d %H:%M')}"
            )
        
        if len(transactions) > 10:
            narrative_parts.append(f"  ... and {len(transactions) - 10} more transactions")
        
        narrative_parts.append("\n--- RISK ASSESSMENT ---")
        narrative_parts.append(f"Overall Risk Score: {alert.priority_score:.2f}")
        narrative_parts.append(f"Confidence Level: {alert.confidence_score:.2f}")
        narrative_parts.append(f"Analysis Agents Triggered: {len(alert.triggered_by)}")
        
        if additional_info:
            narrative_parts.append("\n--- ADDITIONAL INFORMATION ---")
            for key, value in additional_info.items():
                narrative_parts.append(f"{key}: {value}")
        
        narrative_parts.append("\n--- RECOMMENDATION ---")
        narrative_parts.append(
            "Based on the analysis, this activity warrants regulatory reporting. "
            "Further investigation is recommended to determine if criminal activity is involved."
        )
        
        return "\n".join(narrative_parts)
    
    def file_sar(self, sar_id: str, filed_by: str) -> bool:
        """
        Mark SAR as filed with regulatory authority.
        
        Args:
            sar_id: SAR identifier
            filed_by: Name of person filing
            
        Returns:
            Success status
        """
        if sar_id not in self.sars:
            logger.error(f"SAR {sar_id} not found")
            return False
        
        sar = self.sars[sar_id]
        sar.filed = True
        sar.filed_at = datetime.utcnow()
        sar.filed_by = filed_by
        sar.confirmation_number = f"CONF-{uuid.uuid4().hex[:12].upper()}"
        
        logger.info(f"SAR {sar_id} filed by {filed_by}. Confirmation: {sar.confirmation_number}")
        
        return True
    
    def get_sar(self, sar_id: str) -> Optional[SAR]:
        """Retrievand a SAR by ID"""
        return self.sars.get(sar_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get SAR statistics"""
        total = len(self.sars)
        filed = sum(1 for sar in self.sars.values() if sar.filed)
        
        return {
            "total_sars": total,
            "filed": filed,
            "pending_filing": total - filed
        }

