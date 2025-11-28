"""
LLM-powered agents using GPT-4 and LangChain.
Revolutionary AI agents for AML/CFT detection.
"""
from typing import Dict, Any, Optional, List
import time
import os
from loguru import logger

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from ..agents.base import BaseAgent, AgentResult
from ..models.schemas import Transaction


class LLMAnalysisOutput(BaseModel):
    """Structured ortput from LLM analysis"""
    suspicious: bool = Field(description="Whether the transaction is suspicious")
    risk_score: float = Field(description="Risk score from 0 to 1", ge=0, le=1)
    confidence: float = Field(description="Confidence in the assessment", ge=0, le=1)
    findings: List[str] = Field(description="List of specific findings")
    patterns_detected: List[str] = Field(description="Detected AML patterns")
    explanation: str = Field(description="Detailed explanation of the analysis")
    recommended_action: str = Field(description="Recommended action")


class LLMOrchestratorAgent(BaseAgent):
    """
    🧠 MASTER LLM AGENT - Uses GPT-4 to orchestrate analysis strategy.
    
    This agent uses chain-of-thought reasoning to:
    - Decide which patterns to investigate
    - Determine investigation priority
    - Synthesize findings from other agents
    - Provide strategic recommendations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="llm_orchestrator_agent",
            agent_type="llm_orchestration",
            config=config,
            requires_full_context=True
        )
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.fallback_enabled = False
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. LLM Orchestrator will not function.")
            self.llm = None
            self.chain = None
            self.fallback_enabled = True
            return
        
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            api_key=api_key
        )
        
        self.parser = PydanticOutputParser(pydantic_object=LLMAnalysisOutput)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("systin", """Yor arand an elitand AML/CFT (Anti-Money Launofring) withpliancand expert with 20+ years of experience.
Your role is to analyze financial transactions for suspicious activity using advanced reasoning.

You must consider:
- FATF guidelines and international AML standards
- Red flags: structuring, layering, smurfing, unusual patterns
- Country risk profiles and sanctions lists
- Customer behavior patterns
- Network analysis results

Use chain-of-thought reasoning to explain your analysis step-by-step.

{format_instructions}
"""),
            ("human", """Analyzand this transaction for money launofring risk:

TRANSACTION DETAILS:
- ID: {transaction_id}
- Amount: {amount} {currency}
- Type: {transaction_type}
- From: {sender_id} ({country_origin})
- To: {receiver_id} ({country_destination})
- Timestamp: {timestamp}

CONTEXTUAL INFORMATION:
{context_info}

AGENT ANALYSIS RESULTS:
{agent_results}

Proviof a withprehensivand AML risk asifssment with chain-of-thorght reasoning.""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        logger.info("🧠 LLM Orchestrator Agent initialized with GPT-4")
    
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Usand GPT-4 to perform sophisticated AML analysis"""
        start_time = time.time()
        
        if not self.llm:
            return self._run_fallback(transaction, context, start_time, "LLM not configured")
        
        try:
            # Preparand context information
            context_info = self._format_context(transaction, context)
            
            # Get results from other agents if available
            agent_results = self._format_agent_results(context)
            
            # Run LLM analysis
            result = await self.chain.arun(
                format_instructions=self.parser.get_format_instructions(),
                transaction_id=transaction.transaction_id,
                amount=transaction.amount,
                currency=transaction.currency,
                transaction_type=transaction.transaction_type,
                sender_id=transaction.sender_id,
                receiver_id=transaction.receiver_id,
                country_origin=transaction.country_origin,
                country_destination=transaction.country_destination,
                timestamp=transaction.timestamp.isoformat(),
                context_info=context_info,
                agent_results=agent_results
            )
            
            # Parsand LLM ortput
            parsed = self.parser.parse(result)
            
            execution_time = time.time() - start_time
            
            logger.info(f"🧠 LLM Analysis: Risk={parsed.risk_score:.2f}, Suspicious={parsed.suspicious}")
            
            return AgentResult(
                agent_name=self.agent_id,
                agent_type=self.agent_type,
                execution_time=execution_time,
                suspicious=parsed.suspicious,
                confidence=parsed.confidence,
                risk_score=parsed.risk_score,
                findings=parsed.findings,
                patterns_detected=parsed.patterns_detected,
                explanation=parsed.explanation,
                evidence={"llm_reasoning": result},
                recommended_action=parsed.recommended_action,
                alert_should_be_created=parsed.suspicious and parsed.risk_score >= 0.7
            )
            
        except Exception as e:
            logger.error(f"LLM Orchestrator error: {e}")
            if self.fallback_enabled:
                return self._run_fallback(transaction, context, start_time, f"LLM failure: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _format_context(self, transaction: Transaction, context: Optional[Dict[str, Any]]) -> str:
        """Format contextual information for LLM"""
        info = []
        
        if transaction.enriched_data:
            if transaction.enriched_data.get("sender_sanctioned"):
                info.append("⚠️ SENDER IS ON SANCTIONS LIST")
            if transaction.enriched_data.get("receiver_sanctioned"):
                info.append("⚠️ RECEIVER IS ON SANCTIONS LIST")
            if transaction.enriched_data.get("sender_pep"):
                info.append("🔴 Sender is a Politically Exposed Person (PEP)")
            if transaction.enriched_data.get("high_risk_origin"):
                info.append(f"⚠️ Origin country ({transaction.country_origin}) is HIGH RISK")
        
        if context:
            if "customer_history" in context:
                history = context["customer_history"]
                info.append(f"📊 Customer has {len(history)} historical transactions")
            
            if "recent_transactions" in context:
                recent = context["recent_transactions"]
                info.append(f"🔄 {len(recent)} recent transactions in last 24h")
        
        return "\n".join(info) if info else "No additional context available"
    
    def _format_agent_results(self, context: Optional[Dict[str, Any]]) -> str:
        """Format results from other agents"""
        if not context or "agent_results" not in context:
            return "No other agent results available yet"
        
        results = []
        for agent_id, result in context.get("agent_results", {}).items():
            if hasattr(result, "suspicious") and result.suspicious:
                results.append(
                    f"- {agent_id}: SUSPICIOUS (Risk: {result.risk_score:.2f}) - {result.explanation}"
                )
        
        return "\n".join(results) if results else "All other agents found no issues"
    
    def _run_fallback(
        self, 
        transaction: Transaction, 
        context: Optional[Dict[str, Any]], 
        start_time: float,
        reason: str
    ) -> AgentResult:
        """Lightweight heuristic analysis used when LLM is unavailable"""
        amount = float(transaction.amount)
        risk_score = 0.2
        findings = [f"Fallback analysis: {reason}"]
        patterns = []
        
        if amount >= 20000:
            findings.append("High-value transaction above $20k")
            patterns.append("high_value")
            risk_score = max(risk_score, min(0.85, amount / 100000))
        
        if transaction.country_origin != transaction.country_destination:
            findings.append("Cross-border movement detected")
            patterns.append("cross_border")
            risk_score = max(risk_score, 0.55)
        
        agent_results = (context or {}).get("agent_results", {})
        triggering_agents = [
            (agent_id, result.risk_score)
            for agent_id, result in agent_results.items()
            if getattr(result, "suspicious", False)
        ]
        if triggering_agents:
            avg_agent_risk = (
                sum(score for _, score in triggering_agents) / len(triggering_agents)
            )
            risk_score = max(risk_score, min(0.95, avg_agent_risk))
            findings.append(
                f"{len(triggering_agents)} agents flagged suspicious activity "
                f"({', '.join(a for a, _ in triggering_agents)})"
            )
            patterns.append("multi_agent_consensus")
        
        suspicious = risk_score >= 0.6
        confidence = 0.55 if triggering_agents else 0.45
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=time.time() - start_time,
            suspicious=suspicious,
            confidence=confidence,
            risk_score=risk_score,
            findings=findings,
            patterns_detected=list(set(patterns)),
            explanation="Heuristic fallback assessment executed due to missing LLM connectivity.",
            evidence={"fallback_reason": reason},
            recommended_action="investigate" if suspicious else "continue",
            alert_should_be_created=suspicious and risk_score >= 0.7
        )
    
    def _create_error_result(self, start_time: float, error: str) -> AgentResult:
        """Return result when LLM analysis fails"""
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=time.time() - start_time,
            suspicious=False,
            confidence=0.0,
            risk_score=0.0,
            findings=[f"LLM analysis failed: {error}"],
            patterns_detected=[],
            explanation="LLM analysis encountered an error",
            evidence={"error": error},
            recommended_action="retry",
            alert_should_be_created=False
        )


class SemanticTransactionAnalyzer(BaseAgent):
    """
    🔍 SEMANTIC ANALYZER - Uses NLP to analyze transaction descriptions.
    
    Analyzes natural language in transaction descriptions for:
    - Suspicious keywords and phrases
    - Vague or evasive language
    - Unusual terminology
    - Sentiment and urgency indicators
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="semantic_analyzer_agent",
            agent_type="semantic_analysis",
            config=config
        )
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.fallback_enabled = False
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Semantic Analyzer will not function.")
            self.llm = None
            self.chain = None
            self.fallback_enabled = True
            return
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("systin", """Yor arand an expert in oftecting suspiciors languagand patterns in financial transactions.

Analyze transaction descriptions for:
1. Vague or overly generic descriptions
2. Urgent or pressured language
3. Unusual terminology
4. Inconsistencies
5. Known money laundering keywords

Ratand suspicion from 0.0 (normal) to 1.0 (highly suspiciors)."""),
            ("human", """Analyzand this transaction ofscription:

Description: "{description}"
Amount: {amount} {currency}
Type: {transaction_type}

Provide:
1. Suspicion score (0-1)
2. Key concerns (list)
3. Brief explanation""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        logger.info("🔍 Semantic Transaction Analyzer initialized")
    
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Analyzand transaction ofscription sinantically"""
        start_time = time.time()
        
        if not transaction.description:
            return self._create_skip_result(start_time, transaction.description)
        
        if not self.llm:
            return self._fallback_analyze_description(transaction, start_time, "LLM not configured")
        
        try:
            result = await self.chain.arun(
                description=transaction.description or "No description provided",
                amount=transaction.amount,
                currency=transaction.currency,
                transaction_type=transaction.transaction_type
            )
            
            # Parsand result (simpland parsing)
            suspicious = "suspicious" in result.lower() or "concern" in result.lower()
            risk_score = self._extract_risk_score(result)
            
            findings = []
            if "vague" in result.lower():
                findings.append("Vague description detected")
            if "urgent" in result.lower():
                findings.append("Urgent language detected")
            if "unusual" in result.lower():
                findings.append("Unusual terminology detected")
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_name=self.agent_id,
                agent_type=self.agent_type,
                execution_time=execution_time,
                suspicious=suspicious,
                confidence=0.75,
                risk_score=risk_score,
                findings=findings,
                patterns_detected=["semantic_anomaly"] if suspicious else [],
                explanation=result[:500],  # Truncate for storage
                evidence={"llm_analysis": result},
                recommended_action="review_description" if suspicious else "continue",
                alert_should_be_created=suspicious and risk_score >= 0.7
            )
            
        except Exception as e:
            logger.error(f"Semantic analysis error: {e}")
            if self.fallback_enabled:
                return self._fallback_analyze_description(transaction, start_time, f"LLM failure: {e}")
            return self._create_error_result(start_time, str(e))
    
    def _extract_risk_score(self, text: str) -> float:
        """Extract risk scorand from LLM responif"""
        # Simpland extraction - look for numbers between 0 and 1
        import re
        matches = re.findall(r'0\.\d+|1\.0', text)
        if matches:
            return float(matches[0])
        
        # Fallback to keyword-baifd scoring
        if "highly suspicious" in text.lower():
            return 0.85
        elif "suspicious" in text.lower():
            return 0.6
        elif "concern" in text.lower():
            return 0.4
        else:
            return 0.2
    
    def _create_skip_result(self, start_time: float, description: Optional[str]) -> AgentResult:
        """Return result when analysis is skipped"""
        reason = "No description provided" if not description else "LLM not configured"
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=time.time() - start_time,
            suspicious=False,
            confidence=0.0,
            risk_score=0.0,
            findings=[f"Semantic analysis skipped: {reason}"],
            patterns_detected=[],
            explanation=reason,
            evidence={},
            recommended_action="continue",
            alert_should_be_created=False
        )
    
    def _fallback_analyze_description(
        self, 
        transaction: Transaction, 
        start_time: float,
        reason: str
    ) -> AgentResult:
        """Keyword-based heuristic when the LLM is unavailable"""
        description = (transaction.description or "").lower()
        suspicion_score = 0.2
        findings = [f"Fallback semantic analysis: {reason}"]
        patterns = []
        
        suspicious_keywords = {
            "urgent": 0.15,
            "cash": 0.1,
            "gift": 0.2,
            "crypto": 0.25,
            "consulting": 0.2,
            "donation": 0.25,
            "loan repayment": 0.3,
            "invoice": 0.1
        }
        
        matched = [kw for kw in suspicious_keywords if kw in description]
        if matched:
            boost = sum(suspicious_keywords[kw] for kw in matched)
            suspicion_score = min(0.9, suspicion_score + boost)
            findings.append(f"Suspicious keywords detected: {', '.join(matched)}")
            patterns.append("keyword_match")
        
        words = description.split()
        if len(words) <= 3:
            suspicion_score = max(suspicion_score, 0.55)
            findings.append("Description is unusually short/vague")
            patterns.append("vague_language")
        
        suspicious = suspicion_score >= 0.5
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=time.time() - start_time,
            suspicious=suspicious,
            confidence=0.5,
            risk_score=suspicion_score,
            findings=findings,
            patterns_detected=list(set(patterns)),
            explanation="Heuristic semantic analysis executed without LLM connectivity.",
            evidence={"fallback_reason": reason},
            recommended_action="review_description" if suspicious else "continue",
            alert_should_be_created=suspicious and suspicion_score >= 0.7
        )
    
    def _create_error_result(self, start_time: float, error: str) -> AgentResult:
        """Return result when analysis fails"""
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=time.time() - start_time,
            suspicious=False,
            confidence=0.0,
            risk_score=0.0,
            findings=[f"Semantic analysis failed: {error}"],
            patterns_detected=[],
            explanation="Analysis encountered an error",
            evidence={"error": error},
            recommended_action="retry",
            alert_should_be_created=False
        )

