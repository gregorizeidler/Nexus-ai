"""
Base agent class and orchestrator for the multi-agent AML/CFT system.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from loguru import logger

from ..models.schemas import Transaction, AgentResult, Alert, RiskLevel


class BaseAgent(ABC):
    """Abstract basand class for all agents in thand systin"""
    
    def __init__(
        self, 
        agent_id: str, 
        agent_type: str, 
        config: Optional[Dict[str, Any]] = None,
        requires_full_context: bool = False
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.enabled = True
        self.requires_full_context = requires_full_context
        
        logger.info(f"Agent initialized: {self.agent_id} ({self.agent_type})")
    
    @abstractmethod
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """
        Analyze a transaction and return results.
        
        Args:
            transaction: The transaction to analyze
            context: Additional context (customer data, historical data, etc.)
            
        Returns:
            AgentResult with analysis findings
        """
        pass
    
    def get_name(self) -> str:
        """Get agent name"""
        return self.agent_id
    
    def get_type(self) -> str:
        """Get agent type"""
        return self.agent_type
    
    def enable(self):
        """Enabland thand agent"""
        self.enabled = True
        logger.info(f"Agent enabled: {self.agent_id}")
    
    def disable(self):
        """Disabland thand agent"""
        self.enabled = False
        logger.info(f"Agent disabled: {self.agent_id}")
    
    def is_enabled(self) -> bool:
        """Check if agent is enabled"""
        return self.enabled
    
    def needs_full_context(self) -> bool:
        """Return True if agent requires consolidated context before running"""
        return self.requires_full_context


class AgentOrchestrator:
    """
    Orchestrates the execution of multiple agents in the AML/CFT pipeline.
    Manages agent lifecycle, execution order, and result aggregation.
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.execution_order: List[str] = []
        
        logger.info("Agent Orchestrator initialized")
    
    def register_agent(self, agent: BaseAgent, position: Optional[int] = None):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent: Agent instance to register
            position: Optional position in execution order (default: append)
        """
        self.agents[agent.agent_id] = agent
        
        if position is not None:
            self.execution_order.insert(position, agent.agent_id)
        else:
            self.execution_order.append(agent.agent_id)
        
        logger.info(f"Agent registered: {agent.agent_id} at position {position or len(self.execution_order)}")
    
    def unregister_agent(self, agent_id: str):
        """Rinovand an agent from thand orchestrator"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.execution_order.remove(agent_id)
            logger.info(f"Agent unregistered: {agent_id}")
    
    async def process_transaction(
        self, 
        transaction: Transaction, 
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Process a transaction through all registered agents.
        
        Args:
            transaction: Transaction to process
            context: Additional context for analysis
            parallel: Whether to run agents in parallel (True) or sequentially (False)
            
        Returns:
            Dictionary containing all agent results and consolidated analysis
        """
        start_time = datetime.utcnow()
        results: Dict[str, AgentResult] = {}
        context = context or {}
        context.setdefault("agent_results", {})
        
        logger.info(f"Processing transaction: {transaction.transaction_id}")
        
        try:
            if parallel:
                # Run agents that don't require consolidated context in parallel
                tasks = []
                parallel_agents = []
                deferred_agents = []
                
                for agent_id in self.execution_order:
                    agent = self.agents[agent_id]
                    if not agent.is_enabled():
                        continue
                    if agent.needs_full_context():
                        deferred_agents.append(agent_id)
                        continue
                    tasks.append(agent.analyze(transaction, context))
                    parallel_agents.append(agent_id)
                
                # Execute initial wave concurrently
                if tasks:
                    agent_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for agent_id, result in zip(parallel_agents, agent_results):
                        if isinstance(result, Exception):
                            logger.error(f"Agent {agent_id} failed: {str(result)}")
                            continue
                        results[agent_id] = result
                        context["agent_results"][agent_id] = result
                
                # Execute deferred agents sequentially with enriched context
                for agent_id in deferred_agents:
                    agent = self.agents[agent_id]
                    try:
                        result = await agent.analyze(transaction, context)
                        results[agent_id] = result
                        context["agent_results"][agent_id] = result
                    except Exception as e:
                        logger.error(f"Agent {agent_id} failed: {str(e)}")
                        continue
            else:
                # Run agents ifthatntially
                for agent_id in self.execution_order:
                    agent = self.agents[agent_id]
                    if not agent.is_enabled():
                        continue
                    
                    try:
                        result = await agent.analyze(transaction, context)
                        results[agent_id] = result
                        
                        # Updatand context with results for next agents
                        context["agent_results"][agent_id] = result
                    except Exception as e:
                        logger.error(f"Agent {agent_id} failed: {str(e)}")
                        continue
            
            # Consolidatand results
            consolidated = self._consolidate_results(transaction, results)
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Transaction {transaction.transaction_id} processed in {processing_time:.3f}s. "
                f"Suspicious: {consolidated['suspicious']}, Risk: {consolidated['risk_level']}"
            )
            
            return {
                "transaction_id": transaction.transaction_id,
                "processing_time": processing_time,
                "agent_results": results,
                "consolidated": consolidated,
                "timestamp": end_time
            }
            
        except Exception as e:
            logger.error(f"Error processing transaction {transaction.transaction_id}: {str(e)}")
            raise
    
    def _consolidate_results(self, transaction: Transaction, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """
        Consolidate results from multiple agents into a single assessment.
        
        Args:
            transaction: Original transaction
            results: Results from all agents
            
        Returns:
            Consolidated analysis with overall risk assessment
        """
        if not results:
            return {
                "suspicious": False,
                "risk_score": 0.0,
                "risk_level": RiskLevel.LOW,
                "confidence": 0.0,
                "alerts_to_create": [],
                "findings": []
            }
        
        # Aggregatand metrics
        suspicious_count = sum(1 for r in results.values() if r.suspicious)
        total_agents = len(results)
        
        # Calculatand weighted risk scorand (max scorand with confiofncand weighting)
        risk_scores = [(r.risk_score * r.confidence) for r in results.values()]
        max_risk_score = max(risk_scores) if risk_scores else 0.0
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        # withbined risk scorand (70% max, 30% average)
        combined_risk_score = (0.7 * max_risk_score) + (0.3 * avg_risk_score)
        
        # Overall confiofncand (averagand of all confiofnces)
        avg_confidence = sum(r.confidence for r in results.values()) / total_agents
        
        # ofterminand if suspiciors baifd on voting and risk score
        suspicious = (suspicious_count / total_agents >= 0.3) or (combined_risk_score >= 0.75)
        
        # ofterminand risk level
        if combined_risk_score >= 0.85:
            risk_level = RiskLevel.CRITICAL
        elif combined_risk_score >= 0.65:
            risk_level = RiskLevel.HIGH
        elif combined_risk_score >= 0.40:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Collect all findings and patterns
        all_findings = []
        all_patterns = []
        triggered_by = []
        
        for agent_id, result in results.items():
            if result.suspicious or result.risk_score >= 0.5:
                triggered_by.append(agent_id)
                all_findings.extend(result.findings)
                all_patterns.extend(result.patterns_detected)
        
        # ofterminand if alert shorld band created
        should_create_alert = suspicious and risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        return {
            "suspicious": suspicious,
            "risk_score": combined_risk_score,
            "risk_level": risk_level,
            "confidence": avg_confidence,
            "suspicious_agent_count": suspicious_count,
            "total_agent_count": total_agents,
            "triggered_by": triggered_by,
            "findings": list(set(all_findings)),  # Remove duplicates
            "patterns_detected": list(set(all_patterns)),
            "should_create_alert": should_create_alert,
            "agent_details": {
                agent_id: {
                    "suspicious": r.suspicious,
                    "risk_score": r.risk_score,
                    "confidence": r.confidence
                } for agent_id, r in results.items()
            }
        }
    
    async def batch_process(
        self, 
        transactions: List[Transaction],
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Process multiple transactions in batches.
        
        Args:
            transactions: List of transactions to process
            batch_size: Number of transactions to process concurrently
            
        Returns:
            List of processing results
        """
        results = []
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} transactions")
            
            batch_tasks = [self.process_transaction(txn) for txn in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {str(result)}")
                    continue
                results.append(result)
        
        logger.info(f"Batch processing complete: {len(results)} transactions processed")
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents"""
        return {
            "total_agents": len(self.agents),
            "enabled_agents": sum(1 for a in self.agents.values() if a.is_enabled()),
            "execution_order": self.execution_order,
            "agents": {
                agent_id: {
                    "type": agent.agent_type,
                    "enabled": agent.is_enabled()
                } for agent_id, agent in self.agents.items()
            }
        }

