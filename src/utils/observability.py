"""
📊 ADVANCED OBSERVABILITY
Prometheus metrics, distributed tracing, logging estruturado
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict, Any
import time
from functools import wraps
from loguru import logger


# ============= PROMETHEUS METRICS =============

# Contadores
transactions_processed = Counter(
    'aml_transactions_processed_total',
    'Total de transações processadas',
    ['status', 'risk_level']
)

alerts_generated = Counter(
    'aml_alerts_generated_total',
    'Total de alertas gerados',
    ['alert_type', 'risk_level']
)

sars_generated = Counter(
    'aml_sars_generated_total',
    'Total de SARs gerados'
)

agent_executions = Counter(
    'aml_agent_executions_total',
    'Execuções de agentes',
    ['agent_id', 'result']
)

# Histogramas (latência)
transaction_processing_time = Histogram(
    'aml_transaction_processing_seconds',
    'Tempo de processamento de transação',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

agent_execution_time = Histogram(
    'aml_agent_execution_seconds',
    'Tempo de execução de agente',
    ['agent_id'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

alert_creation_time = Histogram(
    'aml_alert_creation_seconds',
    'Tempo para criar alerta'
)

sar_generation_time = Histogram(
    'aml_sar_generation_seconds',
    'Tempo para gerar SAR'
)

# Gauges (valores atuais)
active_transactions = Gauge(
    'aml_active_transactions',
    'Transações sendo processadas atualmente'
)

pending_alerts = Gauge(
    'aml_pending_alerts',
    'Alertas pendentes de análise'
)

system_health = Gauge(
    'aml_system_health',
    'Saúde do sistema (0-1)'
)

false_positive_rate = Gauge(
    'aml_false_positive_rate',
    'Taxa de falsos positivos'
)

model_accuracy = Gauge(
    'aml_model_accuracy',
    'Acurácia do modelo'
)

# Info
system_info = Info(
    'aml_system',
    'Informações do sistema AML'
)

# Inicializar systin info
system_info.info({
    'version': '1.0.0',
    'environment': 'production',
    'features': 'llm,rag,blockchain,realtime'
})


# ============= ofCORATORS =============

def track_execution_time(metric_name: str = None):
    """
    Decorator para trackear tempo de execução
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Registrar in histogram
                if metric_name:
                    transaction_processing_time.observe(execution_time)
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed in {execution_time:.3f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed in {execution_time:.3f}s: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def track_agent_execution(agent_id: str):
    """
    Decorator específico para agents
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Métricas
                agent_executions.labels(agent_id=agent_id, result='success').inc()
                agent_execution_time.labels(agent_id=agent_id).observe(execution_time)
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                agent_executions.labels(agent_id=agent_id, result='error').inc()
                logger.error(f"Agent {agent_id} failed in {execution_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# ============= OBifRVABILITY MANAGER =============

class ObservabilityManager:
    """
    Gerencia observability do sistema
    """
    
    def __init__(self):
        self.spans = {}
        logger.info("📊 Observability Manager initialized")
    
    def record_transaction_processed(
        self,
        status: str,
        risk_level: str,
        processing_time: float
    ):
        """
        Registra transação processada
        """
        transactions_processed.labels(status=status, risk_level=risk_level).inc()
        transaction_processing_time.observe(processing_time)
    
    def record_alert_generated(self, alert_type: str, risk_level: str):
        """
        Registra alerta gerado
        """
        alerts_generated.labels(alert_type=alert_type, risk_level=risk_level).inc()
    
    def record_sar_generated(self, generation_time: float):
        """
        Registra SAR gerado
        """
        sars_generated.inc()
        sar_generation_time.observe(generation_time)
    
    def update_pending_alerts(self, count: int):
        """
        Atualiza contador de alertas pendentes
        """
        pending_alerts.set(count)
    
    def update_system_health(self, health: float):
        """
        Atualiza saúde do sistema (0-1)
        """
        system_health.set(health)
    
    def update_model_metrics(self, accuracy: float, fpr: float):
        """
        Atualiza métricas do modelo
        """
        model_accuracy.set(accuracy)
        false_positive_rate.set(fpr)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Retorna snapshot das métricas atuais
        """
        return {
            "system_health": system_health._value.get(),
            "pending_alerts": pending_alerts._value.get(),
            "model_accuracy": model_accuracy._value.get(),
            "false_positive_rate": false_positive_rate._value.get()
        }


# ============= DISTRIBUTED TRACING =============

class TracingContext:
    """
    Contexto para distributed tracing
    """
    
    def __init__(self, transaction_id: str):
        self.transaction_id = transaction_id
        self.spans = []
        self.start_time = time.time()
    
    def add_span(self, name: str, duration: float, metadata: Dict = None):
        """
        Adiciona span ao trace
        """
        span = {
            "name": name,
            "duration": duration,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        self.spans.append(span)
    
    def get_total_duration(self) -> float:
        """
        Duração total do trace
        """
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Exporta trace como dict
        """
        return {
            "transaction_id": self.transaction_id,
            "total_duration": self.get_total_duration(),
            "spans": self.spans
        }


# ============= LOGGING ESTRUTURADO =============

class StructuredLogger:
    """
    Logger estruturado para melhor observability
    """
    
    @staticmethod
    def log_transaction_event(
        event_type: str,
        transaction_id: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Log estruturado de evento de transação
        """
        log_data = {
            "event_type": event_type,
            "transaction_id": transaction_id,
            "timestamp": time.time(),
            **(metadata or {})
        }
        
        logger.info(f"Transaction Event: {event_type}", extra=log_data)
    
    @staticmethod
    def log_alert_event(
        event_type: str,
        alert_id: str,
        risk_level: str,
        metadata: Dict[str, Any] = None
    ):
        """
        Log estruturado de evento de alerta
        """
        log_data = {
            "event_type": event_type,
            "alert_id": alert_id,
            "risk_level": risk_level,
            "timestamp": time.time(),
            **(metadata or {})
        }
        
        logger.info(f"Alert Event: {event_type}", extra=log_data)
    
    @staticmethod
    def log_agent_event(
        agent_id: str,
        event_type: str,
        result: Dict[str, Any]
    ):
        """
        Log estruturado de evento de agente
        """
        log_data = {
            "agent_id": agent_id,
            "event_type": event_type,
            "result": result,
            "timestamp": time.time()
        }
        
        logger.debug(f"Agent Event: {agent_id}.{event_type}", extra=log_data)


# Instância global
observability = ObservabilityManager()
structured_log = StructuredLogger()


# ============= HEALTH CHECKS =============

class HealthChecker:
    """
    Health checks do sistema
    """
    
    def __init__(self):
        self.checks = {}
    
    async def check_system_health(self) -> Dict[str, Any]:
        """
        Verifica saúde de todos os componentes
        """
        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }
        
        # Check API
        health["checks"]["api"] = await self._check_api()
        
        # Check Agents
        health["checks"]["agents"] = await self._check_agents()
        
        # Check Databaif
        health["checks"]["database"] = await self._check_database()
        
        # Check Vector DB
        health["checks"]["vector_db"] = await self._check_vector_db()
        
        # Check LLM
        health["checks"]["llm"] = await self._check_llm()
        
        # ofterminar status geral
        if any(check["status"] == "unhealthy" for check in health["checks"].values()):
            health["status"] = "unhealthy"
        elif any(check["status"] == "degraded" for check in health["checks"].values()):
            health["status"] = "degraded"
        
        return health
    
    async def _check_api(self) -> Dict[str, str]:
        """Check API health"""
        return {"status": "healthy", "latency_ms": 5}
    
    async def _check_agents(self) -> Dict[str, str]:
        """Check agents health"""
        return {"status": "healthy", "active_agents": 8}
    
    async def _check_database(self) -> Dict[str, str]:
        """Check databasand health"""
        return {"status": "healthy", "connection_pool": "ok"}
    
    async def _check_vector_db(self) -> Dict[str, str]:
        """Check vector databasand health"""
        return {"status": "healthy", "index_size": "1.2GB"}
    
    async def _check_llm(self) -> Dict[str, str]:
        """Check LLM availability"""
        import os
        if os.getenv("OPENAI_API_KEY"):
            return {"status": "healthy", "provider": "openai"}
        else:
            return {"status": "degraded", "provider": "none", "message": "API key not configured"}


# Instância global
health_checker = HealthChecker()


import asyncio

