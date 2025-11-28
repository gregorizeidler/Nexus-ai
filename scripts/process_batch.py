"""
Process a batch of transactions from a JSON file.
"""
import asyncio
import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.schemas import Transaction
from src.agents.base import AgentOrchestrator
from src.agents.ingestion import DataIngestionAgent, EnrichmentAgent, CustomerProfileAgent
from src.agents.analysis import RulesBasedAgent, BehavioralMLAgent, NetworkAnalysisAgent
from src.agents.alert_manager import AlertManager, SARGenerator
from loguru import logger


async def process_batch(transactions_file: str):
    """Process transactions from a file"""
    
    # Initialize system
    logger.info("Initializing AML/CFT system...")
    orchestrator = AgentOrchestrator()
    alert_manager = AlertManager()
    sar_generator = SARGenerator()
    
    # Register agents
    orchestrator.register_agent(DataIngestionAgent())
    orchestrator.register_agent(EnrichmentAgent())
    orchestrator.register_agent(CustomerProfileAgent())
    orchestrator.register_agent(RulesBasedAgent())
    orchestrator.register_agent(BehavioralMLAgent())
    orchestrator.register_agent(NetworkAnalysisAgent())
    
    logger.info(f"System initialized with {len(orchestrator.agents)} agents")
    
    # Load transactions
    logger.info(f"Loading transactions from {transactions_file}...")
    with open(transactions_file, 'r') as f:
        txn_data = json.load(f)
    
    transactions = [Transaction(**txn) for txn in txn_data]
    logger.info(f"Loaded {len(transactions)} transactions")
    
    # Process transactions
    logger.info("Processing transactions...")
    results = await orchestrator.batch_process(transactions, batch_size=10)
    
    # Create alerts
    alerts_created = 0
    sars_generated = 0
    
    for result in results:
        consolidated = result["consolidated"]
        
        if consolidated.get("should_create_alert", False):
            # Find transaction
            txn = next(t for t in transactions if t.transaction_id == result["transaction_id"])
            
            # Create alert
            alert = alert_manager.create_alert(
                transaction=txn,
                agent_results=result["agent_results"],
                consolidated_analysis=consolidated
            )
            alerts_created += 1
            
            # Auto-generate SAR for critical risks
            if alert.risk_level.value == "critical" and alert.priority_score >= 0.90:
                sar = sar_generator.generate_sar(
                    alert=alert,
                    transactions=[txn],
                    additional_info={"auto_generated": True}
                )
                sars_generated += 1
                logger.warning(f"Auto-generated SAR: {sar.sar_id} for critical alert {alert.alert_id}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Transactions processed: {len(results)}")
    logger.info(f"Alerts created: {alerts_created}")
    logger.info(f"SARs auto-generated: {sars_generated}")
    logger.info(f"Average processing time: {sum(r['processing_time'] for r in results) / len(results):.3f}s")
    
    # Alert statistics
    stats = alert_manager.get_statistics()
    logger.info("\nAlert Statistics:")
    logger.info(f"  Total: {stats['total_alerts']}")
    logger.info(f"  By Risk Level: {stats.get('by_risk_level', {})}")
    logger.info(f"  By Type: {stats.get('by_type', {})}")
    
    logger.info("=" * 60)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process transaction batch")
    parser.add_argument("--file", type=str, required=True, help="Path to transactions JSON file")
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        logger.error(f"File not found: {args.file}")
        sys.exit(1)
    
    asyncio.run(process_batch(args.file))


if __name__ == "__main__":
    main()

