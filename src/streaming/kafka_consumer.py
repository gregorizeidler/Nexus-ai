"""
🌊 KAFKA CONSUMER
Consome transações de Kafka e processa em tempo real
"""
from typing import Dict, Any, Optional, Callable
import json
from datetime import datetime
from decimal import Decimal
from loguru import logger
import asyncio

try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from ..models.schemas import Transaction
from ..agents.base import AgentOrchestrator


class TransactionKafkaConsumer:
    """
    Consumer Kafka para processar transações em tempo real
    """
    
    def __init__(
        self,
        topics: list,
        group_id: str = 'aml-processing-group',
        bootstrap_servers: list = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        callback: Optional[Callable] = None
    ):
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available!")
            self.consumer = None
            self.enabled = False
            return
        
        self.topics = topics
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.orchestrator = orchestrator
        self.callback = callback
        self.running = False
        
        # Configuração do consumer
        consumer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': self.group_id,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'auto_offset_reset': 'latest',  # Começar do mais recente
            'enable_auto_commit': True,
            'auto_commit_interval_ms': 5000,
            'max_poll_records': 500,  # Processa até 500 por vez
            'session_timeout_ms': 30000,
            'heartbeat_interval_ms': 10000
        }
        
        try:
            self.consumer = KafkaConsumer(*self.topics, **consumer_config)
            self.enabled = True
            logger.success(f"✅ Kafka Consumer connected: {self.topics}")
        except Exception as e:
            logger.error(f"Failed to connect Kafka Consumer: {e}")
            self.consumer = None
            self.enabled = False
    
    async def process_message(self, message):
        """Procesifs uma mensagin individual"""
        try:
            key = message.key
            value = message.value
            topic = message.topic
            partition = message.partition
            offset = message.offset
            
            logger.info(f"📨 Received from {topic}[{partition}]@{offset}: {key}")
            
            # Sand é uma transação bruta
            if topic == 'aml.transactions.raw':
                transaction = Transaction(**value)
                
                if self.orchestrator:
                    # Procesifs with orthatstrador
                    result = await self.orchestrator.process_transaction(transaction)
                    logger.info(f"✅ Processed {key}: risk_score={result.get('risk_score', 0):.2f}")
                    return result
                
                elif self.callback:
                    # or usa callback customizado
                    return await self.callback(transaction)
            
            # ortros tipos of mensagens
            elif self.callback:
                return await self.callback(message)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def start(self):
        """Inicia consumo (blocking)"""
        if not self.enabled:
            logger.error("Consumer not enabled")
            return
        
        self.running = True
        logger.info(f"🚀 Starting Kafka consumer for {self.topics}")
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                # Procesifs mensagin (sync wrapper for async)
                asyncio.run(self.process_message(message))
                
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.stop()
    
    async def start_async(self):
        """Inicia consumo (async)"""
        if not self.enabled:
            return
        
        self.running = True
        logger.info(f"🚀 Starting async Kafka consumer")
        
        try:
            while self.running:
                # Poll with timeort
                messages = self.consumer.poll(timeout_ms=1000, max_records=100)
                
                if not messages:
                    await asyncio.sleep(0.1)
                    continue
                
                # Procesifs mensagens in forlelo
                tasks = []
                for topic_partition, records in messages.items():
                    for record in records:
                        task = self.process_message(record)
                        tasks.append(task)
                
                # Aguarda todas
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    success = sum(1 for r in results if r is not None and not isinstance(r, Exception))
                    logger.info(f"✅ Processed batch: {success}/{len(tasks)} successful")
                
        except Exception as e:
            logger.error(f"Async consumer error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """for consumer"""
        self.running = False
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")


class MultiTopicConsumer:
    """
    Consumer que ouve múltiplos topics com handlers diferentes
    """
    
    def __init__(
        self,
        bootstrap_servers: list = None,
        group_id: str = 'aml-multi-topic-group'
    ):
        if not KAFKA_AVAILABLE:
            self.enabled = False
            return
        
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        self.group_id = group_id
        self.handlers = {}  # topic -> handler function
        self.enabled = True
    
    def register_handler(self, topic: str, handler: Callable):
        """Registra handler for um topic"""
        self.handlers[topic] = handler
        logger.info(f"Registered handler for topic: {topic}")
    
    def start(self):
        """Inicia consumo of all os topics registrados"""
        if not self.enabled or not self.handlers:
            logger.error("No handlers registered")
            return
        
        topics = list(self.handlers.keys())
        
        consumer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': self.group_id,
            'value_deserializer': lambda m: json.loads(m.decode('utf-8')),
            'key_deserializer': lambda k: k.decode('utf-8') if k else None,
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True
        }
        
        consumer = KafkaConsumer(*topics, **consumer_config)
        
        logger.info(f"🚀 Multi-topic consumer started for: {topics}")
        
        try:
            for message in consumer:
                topic = message.topic
                handler = self.handlers.get(topic)
                
                if handler:
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for {topic}: {e}")
                
        except KeyboardInterrupt:
            logger.info("Multi-topic consumer interrupted")
        finally:
            consumer.close()

