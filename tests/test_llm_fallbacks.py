from decimal import Decimal
from datetime import datetime

import pytest

from src.agents.llm_agents import LLMOrchestratorAgent, SemanticTransactionAnalyzer
from src.models.schemas import AgentResult, Transaction, TransactionType


def _base_transaction(description: str = "urgent cash gift") -> Transaction:
    return Transaction(
        transaction_id="TXN-FALLBACK-1",
        timestamp=datetime.utcnow(),
        amount=Decimal("25000"),
        currency="USD",
        transaction_type=TransactionType.WIRE_TRANSFER,
        sender_id="CUST-X",
        receiver_id="CUST-Y",
        country_origin="US",
        country_destination="MX",
        description=description,
    )


def _agent_result() -> AgentResult:
    return AgentResult(
        agent_name="rules_agent",
        agent_type="rules",
        execution_time=0.001,
        suspicious=True,
        confidence=0.9,
        risk_score=0.82,
        findings=["Test finding"],
        patterns_detected=["pattern"],
        explanation="Unit test",
        evidence={},
        recommended_action="investigate",
        alert_should_be_created=True,
    )


@pytest.mark.asyncio
async def test_llm_orchestrator_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent = LLMOrchestratorAgent()
    transaction = _base_transaction()
    context = {"agent_results": {"rules_agent": _agent_result()}}

    result = await agent.analyze(transaction, context)

    assert result.suspicious
    assert result.evidence.get("fallback_reason")
    assert "agents flagged suspicious activity" in " ".join(result.findings)


@pytest.mark.asyncio
async def test_semantic_analyzer_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent = SemanticTransactionAnalyzer()
    transaction = _base_transaction(description="Urgent consulting cash gift")

    result = await agent.analyze(transaction, context={})

    assert result.suspicious
    assert any("keywords" in finding.lower() for finding in result.findings)

