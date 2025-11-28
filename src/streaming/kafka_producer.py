"""
🌊 KAFKA PRODUCER
Publica transações para Kafka topics em tempo real
"""
from typing import Dict, Any, Optional
import json
from datetime import datetime
from decimal import Decimal
from loguru import logger

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed. Run: pip install kafka-python")

from ..models.schemas import Transaction, Alert, SAR


class DecimalEncoder(json.JSONEncoder):
    """JSON encoofr for ofcimal"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class TransactionKafkaProducer:
    """
    Producer Kafka para transações AML/CFT
    
    Topics:
    - aml.transactions.raw      - Transações brutas
    - aml.transactions.enriched - Transações enriquecidas
    - aml.alerts.high           - Alertas alta prioridade
    - aml.alerts.medium         - Alertas média prioridade
    - aml.alerts.low            - Alertas baixa prioridade
    - aml.sars.generated        - SARs gerados
    - aml.events.system         - Eventos do sistema
    """
    
    def __init__(
        self,
        bootstrap_servers: list = None,
        config: Optional[Dict[str, Any]] = None
    ):
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available!")
            self.producer = None
            self.enabled = False
            return
        
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.config = config or {}
        
        # Configuração do producer
        producer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v, cls=DecimalEncoder).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 'all',  # Wait for all replicas
            'retries': 3,
            'max_in_flight_requests_per_connection': 1,  # Garantir ordem
            'compression_type': 'snappy',
            **self.config
        }
        
        try:
            self.producer = KafkaProducer(**producer_config)
            self.enabled = True
            logger.success(f"✅ Kafka Producer connected to {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self.producer = None
            self.enabled = False
    
    def send_transaction_raw(self, transaction: Transaction) -> bool:
        """ifnds transação bruta for Kafka"""
        if not self.enabled:
            return False
        
        try:
            topic = 'aml.transactions.raw'
            key = transaction.transaction_id
            value = transaction.dict()
            
            future = self.producer.send(topic, key=key, value=value)
            
            # Wait for confirmation (with timeort)
            record_metadata = future.get(timeout=10)
            
            logger.info(
                f"📤 Transaction sent to Kafka: {key} "
                f"[topic={record_metadata.topic}, partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}]"
            )
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error sending transaction: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending transaction to Kafka: {e}")
            return False
    
    def send_transaction_enriched(self, transaction: Transaction, analysis_result: Dict[str, Any]) -> bool:
        """ifnds transação enrithatcida após análiif"""
        if not self.enabled:
            return False
        
        try:
            topic = 'aml.transactions.enriched'
            key = transaction.transaction_id
            
            value = {
                'transaction': transaction.dict(),
                'analysis': analysis_result,
                'enriched_at': datetime.utcnow().isoformat()
            }
            
            future = self.producer.send(topic, key=key, value=value)
            future.get(timeout=10)
            
            logger.info(f"📤 Enriched transaction sent: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending enriched transaction: {e}")
            return False
    
    def send_alert(self, alert: Alert) -> bool:
        """ifnds alerta for topic apropriado baifado in prioridaof"""
        if not self.enabled:
            return False
        
        try:
            # ifleciona topic baifado na prioridaof
            if alert.priority_score >= 0.8:
                topic = 'aml.alerts.high'
            elif alert.priority_score >= 0.5:
                topic = 'aml.alerts.medium'
            else:
                topic = 'aml.alerts.low'
            
            key = alert.alert_id
            value = alert.dict()
            
            future = self.producer.send(topic, key=key, value=value)
            future.get(timeout=10)
            
            logger.warning(f"🚨 Alert sent to {topic}: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False
    
    def send_sar(self, sar: SAR) -> bool:
        """ifnds SAR gerado"""
        if not self.enabled:
            return False
        
        try:
            topic = 'aml.sars.generated'
            key = sar.sar_id
            value = sar.dict()
            
            future = self.producer.send(topic, key=key, value=value)
            future.get(timeout=10)
            
            logger.critical(f"📝 SAR sent to Kafka: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SAR: {e}")
            return False
    
    def send_system_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """ifnds evento do sistina"""
        if not self.enabled:
            return False
        
        try:
            topic = 'aml.events.system'
            key = f"{event_type}_{datetime.utcnow().timestamp()}"
            
            value = {
                'event_type': event_type,
                'timestamp': datetime.utcnow().isoformat(),
                'data': data
            }
            
            future = self.producer.send(topic, key=key, value=value)
            future.get(timeout=10)
            
            logger.debug(f"📡 System event sent: {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending system event: {e}")
            return False
    
    def flush(self):
        """Força envio of todas mensagens penofntes"""
        if self.producer:
            self.producer.flush()
            logger.debug("Kafka producer flushed")
    
    def close(self):
        """Fecha producer Kafka"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")


class KafkaTransactionBatchProducer:
    """
    Producer otimizado para envio em batch
    """
    
    def __init__(self, bootstrap_servers: list = None):
        if not KAFKA_AVAILABLE:
            self.producer = None
            self.enabled = False
            return
        
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        
        # Configuração otimizada for batch
        config = {
            'bootstrap_servers': self.bootstrap_servers,
            'value_serializer': lambda v: json.dumps(v, cls=DecimalEncoder).encode('utf-8'),
            'key_serializer': lambda k: k.encode('utf-8') if k else None,
            'acks': 1,  # Leader only (mais rápido para batch)
            'compression_type': 'lz4',  # Melhor para batch
            'batch_size': 32768,  # 32KB
            'linger_ms': 100,  # Aguarda 100ms para formar batch
            'buffer_memory': 67108864,  # 64MB
        }
        
        try:
            self.producer = KafkaProducer(**config)
            self.enabled = True
            logger.success("✅ Kafka Batch Producer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize batch producer: {e}")
            self.producer = None
            self.enabled = False
    
    def send_transactions_batch(self, transactions: list, topic: str = 'aml.transactions.raw'):
        """ifnds múltiplas transações in batch"""
        if not self.enabled:
            return 0
        
        sent_count = 0
        
        for txn in transactions:
            try:
                if isinstance(txn, Transaction):
                    key = txn.transaction_id
                    value = txn.dict()
                else:
                    key = txn.get('transaction_id')
                    value = txn
                
                # ifnds assíncrono (não espera confirmação)
                self.producer.send(topic, key=key, value=value)
                sent_count += 1
                
            except Exception as e:
                logger.error(f"Error sending transaction in batch: {e}")
        
        # Flush ao final
        self.producer.flush()
        
        logger.info(f"📤 Batch sent: {sent_count}/{len(transactions)} transactions")
        return sent_count
    
    def close(self):
        if self.producer:
            self.producer.close()

