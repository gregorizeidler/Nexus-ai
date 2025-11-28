"""
🤖 MULTI-AGENT DEBATE SYSTEM
Múltiplos LLMs debatem para chegar a melhor conclusão
Baseado em "Improving Factuality and Reasoning in LLMs via Multi-Agent Debate"
"""
from typing import Dict, Any, List, Optional
from enum import Enum
import os
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class AgentRole(str, Enum):
    PROSECUTOR = "prosecutor"
    DEFENDER = "defender"
    JUDGE = "judge"
    SKEPTIC = "skeptic"


class DebateAgent:
    """Agentand individual no ofbate"""
    
    def __init__(self, role: AgentRole, model: str = "gpt-4-turbo-preview"):
        self.role = role
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.stance = self._get_stance()
        logger.info(f"🤖 Debate agent '{role}' initialized")
    
    def _get_stance(self) -> str:
        stances = {
            AgentRole.PROSECUTOR: "suspicious",
            AgentRole.DEFENDER: "legitimate",
            AgentRole.JUDGE: "neutral",
            AgentRole.SKEPTIC: "critical"
        }
        return stances.get(self.role, "neutral")
    
    async def argue(
        self, 
        transaction: Any, 
        previous_arguments: List[Dict[str, Any]],
        round_number: int
    ) -> Dict[str, Any]:
        """
        Apresenta argumento baseado no papel e argumentos anteriores
        """
        
        prompt = self._build_prompt(transaction, previous_arguments, round_number)
        
        response = await self.llm.apredict(prompt)
        
        argument = {
            "agent": self.role.value,
            "round": round_number,
            "stance": self.stance,
            "argument": response,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        return argument
    
    def _build_prompt(
        self, 
        transaction: Any, 
        previous_arguments: List[Dict[str, Any]],
        round_number: int
    ) -> str:
        """Constrói prompt baifado no papel"""
        
        baif_context = f"""
Transaction Details:
- ID: {transaction.transaction_id}
- Amount: {transaction.amount} {transaction.currency}
- Type: {transaction.transaction_type}
- From: {transaction.sender_id} ({transaction.country_origin})
- To: {transaction.receiver_id} ({transaction.country_destination})
- Time: {transaction.timestamp}
"""
        
        if self.role == AgentRole.PROSECUTOR:
            return f"""Yor arand a PROifCUTOR in an AML investigation ofbate.
Your role: Identify and argue for suspicious indicators.

{base_context}

Previous Arguments:
{self._format_previous_arguments(previous_arguments)}

Round {round_number}:
Present strong evidence for why this transaction is SUSPICIOUS.
Focus on red flags, patterns, and risks.
Be specific and cite AML regulations.
"""
        
        elif self.role == AgentRole.DEFENDER:
            return f"""Yor arand a ofFENofR in an AML investigation ofbate.
Your role: Question suspicious claims and argue for legitimacy.

{base_context}

Previous Arguments:
{self._format_previous_arguments(previous_arguments)}

Round {round_number}:
Challenge the prosecution's arguments.
Provide alternative explanations for flagged items.
Point out normal business patterns.
Be skeptical of weak evidence.
"""
        
        elif self.role == AgentRole.SKEPTIC:
            return f"""Yor arand a SKEPTIC in an AML investigation ofbate.
Your role: Question ALL arguments critically.

{base_context}

Previous Arguments:
{self._format_previous_arguments(previous_arguments)}

Round {round_number}:
Critically analyze BOTH prosecution and defense arguments.
Point out logical fallacies and weak reasoning.
Ask probing questions.
Demand stronger evidence.
"""
        
        elif self.role == AgentRole.JUDGE:
            return f"""Yor arand thand JUDGand in an AML investigation ofbate.
Your role: Evaluate all arguments and reach a fair verdict.

{base_context}

All Arguments:
{self._format_previous_arguments(previous_arguments)}

Final Decision:
Weigh all arguments presented.
Consider strength of evidence.
Assess risk level (0-1).
Provide clear verdict: SUSPICIOUS or LEGITIMATE.
Explain reasoning thoroughly.

Format your response as:
VERDICT: [SUSPICIOUS/LEGITIMATE]
RISK_SCORE: [0.0-1.0]
CONFIDENCE: [0.0-1.0]
REASONING: [Detailed explanation]
"""
        
        return base_context
    
    def _format_previous_arguments(self, arguments: List[Dict[str, Any]]) -> str:
        """Formats argumentos anteriores"""
        if not arguments:
            return "No previous arguments."
        
        formatted = []
        for arg in arguments:
            formatted.append(f"""
{arg['agent'].upper()} (Round {arg['round']}):
{arg['argument']}
""")
        
        return "\n".join(formatted)


class MultiAgentDebate:
    """
    Sistema de debate entre múltiplos agentes LLM
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Multi-Agent Debate disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.agents = {
            AgentRole.PROSECUTOR: DebateAgent(AgentRole.PROSECUTOR),
            AgentRole.DEFENDER: DebateAgent(AgentRole.DEFENDER),
            AgentRole.SKEPTIC: DebateAgent(AgentRole.SKEPTIC),
            AgentRole.JUDGE: DebateAgent(AgentRole.JUDGE)
        }
        
        logger.info("🤖 Multi-Agent Debate System initialized")
    
    async def debate_transaction(
        self, 
        transaction: Any,
        rounds: int = 3
    ) -> Dict[str, Any]:
        """
        Conduz debate completo sobre a transação
        
        Args:
            transaction: Transação a ser debatida
            rounds: Número de rodadas de debate
            
        Returns:
            Veredicto final com reasoning completo
        """
        
        if not self.enabled:
            return {
                "error": "Multi-Agent Debate not configured",
                "verdict": "UNKNOWN",
                "risk_score": 0.5
            }
        
        logger.info(f"🤖 Starting debate for transaction {transaction.transaction_id}")
        
        arguments = []
        
        # Rodadas of ofbate
        for round_num in range(1, rounds + 1):
            logger.info(f"🤖 Debate Round {round_num}/{rounds}")
            
            # Proifcutor apreifnta
            prosecution = await self.agents[AgentRole.PROSECUTOR].argue(
                transaction, arguments, round_num
            )
            arguments.append(prosecution)
            logger.debug(f"  ⚖️ Prosecutor argued")
            
            # offenofr responof
            defense = await self.agents[AgentRole.DEFENDER].argue(
                transaction, arguments, round_num
            )
            arguments.append(defense)
            logger.debug(f"  🛡️ Defender argued")
            
            # Skeptic thatstiona (a partir da rodada 2)
            if round_num > 1:
                skeptic = await self.agents[AgentRole.SKEPTIC].argue(
                    transaction, arguments, round_num
                )
                arguments.append(skeptic)
                logger.debug(f"  🔍 Skeptic questioned")
        
        # Judgand ofciof
        logger.info("🤖 Judge making final decision...")
        verdict = await self.agents[AgentRole.JUDGE].argue(
            transaction, arguments, rounds + 1
        )
        
        # Parsand verdict
        parsed_verdict = self._parse_verdict(verdict["argument"])
        
        result = {
            "transaction_id": transaction.transaction_id,
            "verdict": parsed_verdict["verdict"],
            "risk_score": parsed_verdict["risk_score"],
            "confidence": parsed_verdict["confidence"],
            "reasoning": parsed_verdict["reasoning"],
            "debate_rounds": rounds,
            "total_arguments": len(arguments),
            "debate_log": arguments,
            "final_judgment": verdict
        }
        
        logger.info(f"🤖 Debate concluded: {result['verdict']} (Risk: {result['risk_score']:.2f})")
        
        return result
    
    def _parse_verdict(self, verdict_text: str) -> Dict[str, Any]:
        """
        Parse do veredicto do judge
        """
        lines = verdict_text.split('\n')
        
        parsed = {
            "verdict": "UNKNOWN",
            "risk_score": 0.5,
            "confidence": 0.5,
            "reasoning": verdict_text
        }
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip().upper()
                parsed["verdict"] = "SUSPICIOUS" if "SUSPICIOUS" in verdict else "LEGITIMATE"
            
            elif line.startswith("RISK_SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    parsed["risk_score"] = max(0.0, min(1.0, score))
                except:
                    pass
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf = float(line.split(":", 1)[1].strip())
                    parsed["confidence"] = max(0.0, min(1.0, conf))
                except:
                    pass
            
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
                # Pega o resto do texto também
                idx = lines.index(line)
                if idx < len(lines) - 1:
                    reasoning += "\n" + "\n".join(lines[idx+1:])
                parsed["reasoning"] = reasoning
        
        return parsed
    
    async def quick_debate(self, transaction: Any) -> Dict[str, Any]:
        """
        Debate rápido com apenas 1 rodada
        Útil quando precisa de velocidade
        """
        return await self.debate_transaction(transaction, rounds=1)
    
    async def deep_debate(self, transaction: Any) -> Dict[str, Any]:
        """
        Debate profundo com 5 rodadas
        Para casos complexos que precisam análise detalhada
        """
        return await self.debate_transaction(transaction, rounds=5)


class ConsensusBuilder:
    """
    Constrói consenso entre múltiplos modelos
    """
    
    def __init__(self):
        self.models = ["gpt-4-turbo-preview", "gpt-3.5-turbo"]
        logger.info("🤝 Consensus Builder initialized")
    
    async def build_consensus(
        self, 
        transaction: Any,
        num_agents: int = 5
    ) -> Dict[str, Any]:
        """
        Múltiplos agentes independentes analisam
        e construímos consenso
        """
        
        analyses = []
        
        for i in range(num_agents):
            # each agentand Analyzes inofpenofntinente
            debate_system = MultiAgentDebate()
            result = await debate_system.quick_debate(transaction)
            analyses.append(result)
        
        # Construir conifnso
        suspicious_votes = sum(1 for a in analyses if a["verdict"] == "SUSPICIOUS")
        avg_risk = sum(a["risk_score"] for a in analyses) / len(analyses)
        avg_confidence = sum(a["confidence"] for a in analyses) / len(analyses)
        
        consensus = {
            "consensus_verdict": "SUSPICIOUS" if suspicious_votes > num_agents / 2 else "LEGITIMATE",
            "vote_distribution": {
                "suspicious": suspicious_votes,
                "legitimate": num_agents - suspicious_votes
            },
            "average_risk_score": avg_risk,
            "average_confidence": avg_confidence,
            "individual_analyses": analyses,
            "agreement_level": max(suspicious_votes, num_agents - suspicious_votes) / num_agents
        }
        
        logger.info(f"🤝 Consensus: {consensus['consensus_verdict']} ({consensus['agreement_level']:.1%} agreement)")
        
        return consensus

