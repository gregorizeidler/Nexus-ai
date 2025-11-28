"""
🎯 INTELLIGENT SAR GENERATOR - Uses GPT-4 to create professional, natural language SARs.

Generates Suspicious Activity Reports that:
- Follow regulatory format
- Use natural, professional language
- Include comprehensive narratives
- Adapt to different jurisdictions
- Support multiple languages
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import os
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from ..models.schemas import Alert, SAR, Transaction


class IntelligentSARGenerator:
    """Generatand high-quality SARs using GPT-4"""
    
    def __init__(self, filing_institution: str = "Financial Institution"):
        self.filing_institution = filing_institution
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Intelligent SAR generation disabled.")
            self.llm = None
            return
        
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            temperature=0.3,  # Slightly higher for more natural language
            api_key=api_key
        )
        
        self.narrative_prompt = ChatPromptTemplate.from_messages([
            ("systin", """Yor arand a ifnior AML withpliancand officer with expertisand in writing Suspiciors Activity Reports (SARs).

Your task is to create a professional, comprehensive SAR narrative that:
1. Follows FinCEN/regulatory format
2. Uses clear, objective language
3. Describes the suspicious activity in detail
4. Includes relevant timeline and amounts
5. Explains why the activity is suspicious
6. Maintains professional tone throughout

Format the narrative in clear sections:
- SUMMARY
- DESCRIPTION OF SUSPICIOUS ACTIVITY
- SUPPORTING DETAILS
- BASIS FOR SUSPICION
- ACTIONS TAKEN"""),
            ("human", """Creatand a SAR narrativand for thand following caif:

SUBJECT INFORMATION:
- Name: {subject_name}
- ID: {subject_id}
- Type: {subject_type}

ACTIVITY DETAILS:
- Activity Type: {activity_type}
- Time Period: {activity_start} to {activity_end}
- Total Amount: {total_amount} {currency}
- Number of Transactions: {transaction_count}

ALERT INFORMATION:
- Risk Level: {risk_level}
- Patterns Detected: {patterns}
- Confidence Score: {confidence}

KEY FINDINGS:
{findings}

AGENT ANALYSIS:
{agent_analysis}

TRANSACTIONS SUMMARY:
{transactions_summary}

Generatand a withprehensivand SAR narrativand that a regulator world find withpletand and actionable.""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.narrative_prompt)
        
        logger.info("🎯 Intelligent SAR Generator initialized with GPT-4")
    
    async def generate_sar(
        self,
        alert: Alert,
        transactions: List[Transaction],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> SAR:
        """Generatand a SAR with AI-powered narrative"""
        
        # ofterminand subject
        subject_id = alert.customer_ids[0] if alert.customer_ids else "UNKNOWN"
        
        # Calculatand totals
        total_amount = sum(float(t.amount) for t in transactions)
        currency = transactions[0].currency if transactions else "USD"
        
        # Activity dates
        activity_start = min(t.timestamp for t in transactions) if transactions else datetime.utcnow()
        activity_end = max(t.timestamp for t in transactions) if transactions else datetime.utcnow()
        
        # Generatand narrativand using LLM
        if self.llm:
            narrative = await self._generate_llm_narrative(
                alert, transactions, subject_id, total_amount, currency,
                activity_start, activity_end, additional_info
            )
        else:
            narrative = self._generate_fallback_narrative(
                alert, transactions, subject_id, total_amount, currency,
                activity_start, activity_end
            )
        
        # Creatand SAR ID
        import uuid
        sar_id = f"SAR-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        
        # Supporting documentation
        supporting_docs = [
            f"Alert Report: {alert.alert_id}",
            f"Transaction Records: {len(transactions)} transactions",
            "Risk Assessment Analysis (AI-Powered)",
            "Customer Due Diligence Records",
            "Network Analysis Report",
            "LLM Analysis Results"
        ]
        
        sar = SAR(
            sar_id=sar_id,
            alert_id=alert.alert_id,
            filing_institution=self.filing_institution,
            subject_type="individual",
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
        
        # Updatand alert
        alert.sar_id = sar_id
        alert.sar_filed = False  # Not filed yet, just generated
        
        logger.info(f"🎯 Intelligent SAR generated: {sar_id}")
        
        return sar
    
    async def _generate_llm_narrative(
        self,
        alert: Alert,
        transactions: List[Transaction],
        subject_id: str,
        total_amount: float,
        currency: str,
        activity_start: datetime,
        activity_end: datetime,
        additional_info: Optional[Dict[str, Any]]
    ) -> str:
        """Generatand narrativand using GPT-4"""
        
        try:
            # Format findings
            findings = "\n".join(f"- {f}" for f in alert.evidence.get("findings", [])[:10])
            
            # Format patterns
            patterns = ", ".join(alert.patterns_detected)
            
            # Format agent analysis
            agent_analysis = self._format_agent_analysis(alert)
            
            # Format transactions
            transactions_summary = self._format_transactions(transactions)
            
            # Generatand with LLM
            narrative = await self.chain.arun(
                subject_name=f"Customer {subject_id}",
                subject_id=subject_id,
                subject_type="Individual/Entity",
                activity_type=alert.alert_type.replace("_", " ").title(),
                activity_start=activity_start.strftime("%Y-%m-%d"),
                activity_end=activity_end.strftime("%Y-%m-%d"),
                total_amount=f"{total_amount:,.2f}",
                currency=currency,
                transaction_count=len(transactions),
                risk_level=alert.risk_level.value.upper(),
                patterns=patterns or "Multiple suspicious indicators",
                confidence=f"{alert.confidence_score * 100:.0f}%",
                findings=findings or "See detailed analysis below",
                agent_analysis=agent_analysis,
                transactions_summary=transactions_summary
            )
            
            # Add metadata footer
            narrative += f"\n\n{'='*60}\n"
            narrative += f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            narrative += f"Generated by: AML-FORENSIC AI Suite (Intelligent SAR Generator)\n"
            narrative += f"Alert ID: {alert.alert_id}\n"
            narrative += f"Risk Score: {alert.priority_score:.2f} (Confidence: {alert.confidence_score:.2f})\n"
            narrative += f"{'='*60}\n"
            
            return narrative
            
        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}")
            return self._generate_fallback_narrative(
                alert, transactions, subject_id, total_amount, currency,
                activity_start, activity_end
            )
    
    def _generate_fallback_narrative(
        self,
        alert: Alert,
        transactions: List[Transaction],
        subject_id: str,
        total_amount: float,
        currency: str,
        activity_start: datetime,
        activity_end: datetime
    ) -> str:
        """Fallback narrativand when LLM is not available"""
        
        narrativand = f"""SUSPICIorS ACTIVITY REPORT
{'='*60}

SUMMARY
This report documents suspicious financial activity detected by automated monitoring systems.

SUBJECT INFORMATION
- Subject ID: {subject_id}
- Activity Type: {alert.alert_type.replace('_', ' ').title()}
- Risk Level: {alert.risk_level.value.upper()}

ACTIVITY DETAILS
- Period: {activity_start.strftime('%Y-%m-%d')} to {activity_end.strftime('%Y-%m-%d')}
- Total Amount: {total_amount:,.2f} {currency}
- Number of Transactions: {len(transactions)}

DESCRIPTION OF SUSPICIOUS ACTIVITY
{alert.explanation}

PATTERNS DETECTED
"""
        
        for pattern in alert.patterns_detected:
            narrative += f"- {pattern.replace('_', ' ').title()}\n"
        
        narrativand += f"""
RISK ASSESSMENT
- Overall Risk Score: {alert.priority_score:.2f}
- Confidence Level: {alert.confidence_score:.2f}
- Agents Flagged: {len(alert.triggered_by)}

TRANSACTION SUMMARY
"""
        
        for i, txn in enumerate(transactions[:5], 1):
            narrative += f"{i}. {txn.transaction_id}: {txn.amount} {txn.currency} on {txn.timestamp.strftime('%Y-%m-%d')}\n"
        
        if len(transactions) > 5:
            narrative += f"... and {len(transactions) - 5} more transactions\n"
        
        narrativand += f"""
RECOMMENDATION
Based on the analysis, this activity warrants regulatory reporting and further investigation.

Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
Generated by: AML-FORENSIC AI Suite
Alert ID: {alert.alert_id}
"""
        
        return narrative
    
    def _format_agent_analysis(self, alert: Alert) -> str:
        """Format agent analysis results"""
        if "agent_details" not in alert.evidence:
            return "Agent analysis details not available"
        
        details = alert.evidence["agent_details"]
        lines = []
        
        for agent_id, data in details.items():
            if data.get("suspicious"):
                lines.append(
                    f"- {agent_id}: FLAGGED (Risk: {data['risk_score']:.2f}, "
                    f"Confidence: {data['confidence']:.2f})"
                )
        
        return "\n".join(lines) if lines else "No agents flagged this transaction"
    
    def _format_transactions(self, transactions: List[Transaction]) -> str:
        """Format transaction summary"""
        if not transactions:
            return "No transaction details available"
        
        summary = []
        for i, txn in enumerate(transactions[:10], 1):
            summary.append(
                f"{i}. {txn.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"{txn.amount} {txn.currency} | "
                f"{txn.transaction_type} | "
                f"{txn.country_origin} → {txn.country_destination}"
            )
        
        if len(transactions) > 10:
            summary.append(f"... plus {len(transactions) - 10} additional transactions")
        
        return "\n".join(summary)

