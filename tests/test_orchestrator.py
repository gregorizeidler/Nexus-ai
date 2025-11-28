import pytest
from decimal import Decimal
from datetime import datetime

from src.agents.base import BaseAgent, AgentOrchestrator
from src.models.schemas import AgentResult, Transaction, TransactionType


class AlwaysSuspiciousAgent(BaseAgent):
    def __init__(self):
        super().__init__("rules_agent", "rules_analysis")

    async def analyze(self, transaction: Transaction, context=None) -> AgentResult:
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=0.001,
            suspicious=True,
            confidence=0.9,
            risk_score=0.85,
            findings=["Test suspicious activity"],
            patterns_detected=["test_pattern"],
            explanation="Unit test agent raised suspicion.",
            evidence={},
            recommended_action="investigate",
            alert_should_be_created=True,
        )


class NeedsContextAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            "llm_proxy_agent",
            "llm_orchestration",
            requires_full_context=True,
        )

    async def analyze(self, transaction: Transaction, context=None) -> AgentResult:
        assert context is not None
        assert "agent_results" in context
        assert "rules_agent" in context["agent_results"]

        contributing_agent = context["agent_results"]["rules_agent"]
        explanation = (
            "Observed upstream agent "
            f"{contributing_agent.agent_name} indicating suspicious activity."
        )

        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=0.001,
            suspicious=True,
            confidence=0.8,
            risk_score=contributing_agent.risk_score,
            findings=["Full-context agent executed after initial wave"],
            patterns_detected=["contextual_reasoning"],
            explanation=explanation,
            evidence={},
            recommended_action="investigate",
            alert_should_be_created=True,
        )


def _sample_transaction() -> Transaction:
    return Transaction(
        transaction_id="TXN-TEST-1",
        timestamp=datetime.utcnow(),
        amount=Decimal("15000.00"),
        currency="USD",
        transaction_type=TransactionType.WIRE_TRANSFER,
        sender_id="CUST-A",
        receiver_id="CUST-B",
        country_origin="US",
        country_destination="BR",
    )


@pytest.mark.asyncio
async def test_orchestrator_shares_context_in_parallel():
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(AlwaysSuspiciousAgent())
    orchestrator.register_agent(NeedsContextAgent())

    transaction = _sample_transaction()
    result = await orchestrator.process_transaction(transaction, parallel=True)

    assert "llm_proxy_agent" in result["agent_results"]
    llm_result = result["agent_results"]["llm_proxy_agent"]
    assert llm_result.suspicious
    assert "Full-context agent executed" in llm_result.findings[0]

