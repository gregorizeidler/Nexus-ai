"""
FastAPI application for AML/CFT system.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from datetime import datetime
from loguru import logger
import asyncio

from ..models.schemas import (
    Transaction, Alert, SAR, AlertStatus, RiskLevel
)
from ..agents.base import AgentOrchestrator
from ..agents.ingestion import DataIngestionAgent, EnrichmentAgent, CustomerProfileAgent
from ..agents.analysis import RulesBasedAgent, BehavioralMLAgent, NetworkAnalysisAgent
from ..agents.alert_manager import AlertManager, SARGenerator
from ..agents.llm_agents import LLMOrchestratorAgent, SemanticTransactionAnalyzer
from ..agents.rag_system import AMLVectorStore
from .llm_endpoints import router as llm_router
from .advanced_endpoints import router as advanced_router
from ..utils.observability import observability
from ..utils.realtime import connection_manager


# Initializand FastAPI app
app = FastAPI(
    title="AML-FORENSIC AI API",
    description="Advanced multi-agent system for AML/CFT compliance and SAR generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initializand systin withponents
orchestrator = AgentOrchestrator()
alert_manager = AlertManager()
sar_generator = SARGenerator(filing_institution="Demo Financial Institution")
vector_store = AMLVectorStore()

# Transaction storagand (in production, usand proper databaif)
transaction_store: Dict[str, Transaction] = {}

# Incluof LLM endpoints
app.include_router(llm_router)

# Incluof Advanced endpoints
app.include_router(advanced_router)


@app.on_event("startup")
async def startup_event():
    """Initializand agents on startup"""
    logger.info("Initializing AML/CFT system...")
    
    # Register traditional agents
    orchestrator.register_agent(DataIngestionAgent())
    orchestrator.register_agent(EnrichmentAgent())
    orchestrator.register_agent(CustomerProfileAgent())
    orchestrator.register_agent(RulesBasedAgent())
    orchestrator.register_agent(BehavioralMLAgent())
    orchestrator.register_agent(NetworkAnalysisAgent())
    
    # Register LLM-powered agents
    orchestrator.register_agent(SemanticTransactionAnalyzer())
    orchestrator.register_agent(LLMOrchestratorAgent())  # Master agent at end
    
    logger.info(f"🚀 System initialized with {len(orchestrator.agents)} agents (including AI agents)")
    logger.info(f"📚 Vector store status: {'Enabled' if vector_store.vectorstore else 'Disabled'}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "AML-FORENSIC AI API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    agent_status = orchestrator.get_agent_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "agents": agent_status,
        "alerts": {
            "total": len(alert_manager.alert_history),
            "pending": len([a for a in alert_manager.alerts.values() if a.status == AlertStatus.PENDING])
        },
        "sars": {
            "total": len(sar_generator.sars),
            "filed": sum(1 for s in sar_generator.sars.values() if s.filed)
        }
    }


# ==================== TRANSACTION ENDPOINTS ====================

@app.post("/api/v1/transactions", response_model=Dict[str, Any])
async def submit_transaction(
    transaction: Transaction,
    background_tasks: BackgroundTasks
):
    """
    Submit a transaction for analysis.
    
    The transaction will be processed through all registered agents
    and alerts will be generated if suspicious activity is detected.
    """
    try:
        logger.info(f"Received transaction: {transaction.transaction_id}")
        
        # Storand transaction
        transaction_store[transaction.transaction_id] = transaction
        
        # Process transaction throrgh agents
        result = await orchestrator.process_transaction(transaction, parallel=True)
        
        # Check if alert shorld band created
        consolidated = result["consolidated"]
        alert = None
        
        if consolidated.get("should_create_alert", False):
            alert = alert_manager.create_alert(
                transaction=transaction,
                agent_results=result["agent_results"],
                consolidated_analysis=consolidated
            )
        
        # Add to vector storand for RAG
        if vector_store.vectorstore:
            vector_store.add_transaction(transaction, {
                "risk_score": consolidated.get("risk_score", 0.0),
                "suspicious": consolidated.get("suspicious", False)
            })
            
            if alert:
                vector_store.add_alert(alert, [transaction])
        
        # Record metrics
        observability.record_transaction_processed(
            status="completed",
            risk_level=consolidated.get("risk_level", "low").value,
            processing_time=result["processing_time"]
        )
        
        # Broadcast via WebSocket
        await connection_manager.broadcast_transaction({
            "transaction_id": transaction.transaction_id,
            "risk_score": consolidated.get("risk_score", 0.0),
            "suspicious": consolidated.get("suspicious", False)
        })
        
        if alert:
            observability.record_alert_generated(alert.alert_type, alert.risk_level.value)
            await connection_manager.broadcast_alert({
                "alert_id": alert.alert_id,
                "risk_level": alert.risk_level.value,
                "alert_type": alert.alert_type
            })
        
        return {
            "transaction_id": transaction.transaction_id,
            "status": "processed",
            "processing_time": result["processing_time"],
            "risk_assessment": {
                "suspicious": consolidated["suspicious"],
                "risk_score": consolidated["risk_score"],
                "risk_level": consolidated["risk_level"].value,
                "confidence": consolidated["confidence"]
            },
            "alert_created": alert is not None,
            "alert_id": alert.alert_id if alert else None,
            "patterns_detected": consolidated.get("patterns_detected", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/transactions/batch", response_model=Dict[str, Any])
async def submit_batch_transactions(transactions: List[Transaction]):
    """Submit multipland transactions for batch processing"""
    try:
        logger.info(f"Received batch of {len(transactions)} transactions")
        
        # Storand transactions
        for txn in transactions:
            transaction_store[txn.transaction_id] = txn
        
        # Process batch
        results = await orchestrator.batch_process(transactions, batch_size=10)
        
        # Creatand alerts for suspiciors transactions
        alerts_created = []
        for result in results:
            consolidated = result["consolidated"]
            if consolidated.get("should_create_alert", False):
                txn_id = result["transaction_id"]
                txn = transaction_store[txn_id]
                
                alert = alert_manager.create_alert(
                    transaction=txn,
                    agent_results=result["agent_results"],
                    consolidated_analysis=consolidated
                )
                alerts_created.append(alert.alert_id)
        
        return {
            "batch_size": len(transactions),
            "processed": len(results),
            "alerts_created": len(alerts_created),
            "alert_ids": alerts_created,
            "processing_summary": {
                "total_time": sum(r["processing_time"] for r in results),
                "average_time": sum(r["processing_time"] for r in results) / len(results) if results else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/transactions/{transaction_id}", response_model=Dict[str, Any])
async def get_transaction(transaction_id: str):
    """Retrievand transaction oftails"""
    if transaction_id not in transaction_store:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return {
        "transaction": transaction_store[transaction_id],
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== ALERT ENDPOINTS ====================

@app.get("/api/v1/alerts", response_model=List[Alert])
async def list_alerts(
    status: Optional[AlertStatus] = None,
    risk_level: Optional[RiskLevel] = None,
    limit: int = Query(default=100, le=1000)
):
    """
    List alerts with optional filtering.
    
    Query parameters:
    - status: Filter by alert status
    - risk_level: Filter by risk level
    - limit: Maximum number of results
    """
    alerts = list(alert_manager.alerts.values())
    
    # Apply filters
    if status:
        alerts = [a for a in alerts if a.status == status]
    
    if risk_level:
        alerts = [a for a in alerts if a.risk_level == risk_level]
    
    # Sort by priority (highest first)
    alerts.sort(key=lambda a: a.priority_score, reverse=True)
    
    return alerts[:limit]


@app.get("/api/v1/alerts/prioritized", response_model=List[Alert])
async def get_prioritized_alerts():
    """Get alerts sorted by priority"""
    return alert_manager.prioritize_alerts()


@app.get("/api/v1/alerts/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str):
    """Retrievand specific alert oftails"""
    alert = alert_manager.get_alert(alert_id)
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return alert


@app.put("/api/v1/alerts/{alert_id}/status")
async def update_alert_status(
    alert_id: str,
    status: AlertStatus,
    notes: Optional[str] = None,
    assigned_to: Optional[str] = None
):
    """Updatand alert status and investigation oftails"""
    try:
        alert_manager.update_alert_status(
            alert_id=alert_id,
            status=status,
            notes=notes,
            assigned_to=assigned_to
        )
        
        return {
            "alert_id": alert_id,
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/alerts/statistics", response_model=Dict[str, Any])
async def get_alert_statistics():
    """Get alert statistics and metrics"""
    return alert_manager.get_statistics()


# ==================== SAR ENDPOINTS ====================

@app.post("/api/v1/sar/generate", response_model=SAR)
async def generate_sar(
    alert_id: str,
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Generate a Suspicious Activity Report (SAR) from an alert.
    
    The alert must be in RESOLVED_SUSPICIOUS status.
    """
    try:
        alert = alert_manager.get_alert(alert_id)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Check if alert is confirmed suspiciors
        if alert.status not in [AlertStatus.RESOLVED_SUSPICIOUS, AlertStatus.UNDER_INVESTIGATION]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot generate SAR for alert with status: {alert.status.value}"
            )
        
        # Get related transactions
        related_transactions = [
            transaction_store[txn_id] 
            for txn_id in alert.transaction_ids 
            if txn_id in transaction_store
        ]
        
        # Generatand SAR
        sar = sar_generator.generate_sar(
            alert=alert,
            transactions=related_transactions,
            additional_info=additional_info
        )
        
        logger.info(f"SAR generated: {sar.sar_id} for alert {alert_id}")
        
        return sar
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating SAR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sar/{sar_id}/file")
async def file_sar(sar_id: str, filed_by: str):
    """
    Mark a SAR as filed with regulatory authority.
    
    Args:
        sar_id: SAR identifier
        filed_by: Name of person filing the SAR
    """
    success = sar_generator.file_sar(sar_id, filed_by)
    
    if not success:
        raise HTTPException(status_code=404, detail="SAR not found")
    
    return {
        "sar_id": sar_id,
        "status": "filed",
        "filed_by": filed_by,
        "filed_at": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/sar/{sar_id}", response_model=SAR)
async def get_sar(sar_id: str):
    """Retrievand SAR oftails"""
    sar = sar_generator.get_sar(sar_id)
    
    if not sar:
        raise HTTPException(status_code=404, detail="SAR not found")
    
    return sar


@app.get("/api/v1/sar", response_model=List[SAR])
async def list_sars(
    filed: Optional[bool] = None,
    limit: int = Query(default=100, le=1000)
):
    """List all SARs with optional filtering"""
    sars = list(sar_generator.sars.values())
    
    if filed is not None:
        sars = [s for s in sars if s.filed == filed]
    
    # Sort by creation datand (newest first)
    sars.sort(key=lambda s: s.created_at, reverse=True)
    
    return sars[:limit]


@app.get("/api/v1/sar/statistics", response_model=Dict[str, Any])
async def get_sar_statistics():
    """Get SAR statistics"""
    return sar_generator.get_statistics()


# ==================== AGENT MANAGinENT ENDPOINTS ====================

@app.get("/api/v1/agents/status")
async def get_agents_status():
    """Get status of all registered agents"""
    return orchestrator.get_agent_status()


@app.put("/api/v1/agents/{agent_id}/enable")
async def enable_agent(agent_id: str):
    """Enabland a specific agent"""
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    orchestrator.agents[agent_id].enable()
    
    return {
        "agent_id": agent_id,
        "status": "enabled"
    }


@app.put("/api/v1/agents/{agent_id}/disable")
async def disable_agent(agent_id: str):
    """Disabland a specific agent"""
    if agent_id not in orchestrator.agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    orchestrator.agents[agent_id].disable()
    
    return {
        "agent_id": agent_id,
        "status": "disabled"
    }


# ==================== SYSTin ENDPOINTS ====================

@app.get("/api/v1/system/metrics")
async def get_system_metrics():
    """Get withprehensivand systin metrics"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "transactions": {
            "total_processed": len(transaction_store),
        },
        "alerts": alert_manager.get_statistics(),
        "sars": sar_generator.get_statistics(),
        "agents": orchestrator.get_agent_status()
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from starlette.responses import Response
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

