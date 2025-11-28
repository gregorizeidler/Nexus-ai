"""
NEXUS AI - Prometheus Metrics Exporter

Exposes custom metrics for monitoring ML models, inference, and business KPIs.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, Info, generate_latest, REGISTRY
from prometheus_client import start_http_server
import time
import random
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL PERFORMANCE METRICS
# ============================================================================

# Model accuracy metrics
model_accuracy = Gauge(
    'nexus_model_accuracy',
    'Current model accuracy score',
    ['model_name', 'environment']
)

model_precision = Gauge(
    'nexus_model_precision',
    'Model precision score',
    ['model_name']
)

model_recall = Gauge(
    'nexus_model_recall',
    'Model recall score',
    ['model_name']
)

model_f1_score = Gauge(
    'nexus_model_f1_score',
    'Model F1 score',
    ['model_name']
)

model_roc_auc = Gauge(
    'nexus_model_roc_auc',
    'Model ROC-AUC score',
    ['model_name']
)

false_positive_rate = Gauge(
    'nexus_false_positive_rate',
    'False positive rate',
    ['model_name']
)

false_negative_rate = Gauge(
    'nexus_false_negative_rate',
    'False negative rate',
    ['model_name']
)


# ============================================================================
# INFERENCE METRICS
# ============================================================================

# Inference latency
inference_duration = Histogram(
    'nexus_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'endpoint'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

# Inference throughput
inference_requests_total = Counter(
    'nexus_inference_requests_total',
    'Total number of inference requests',
    ['model_name', 'status']
)

# Batch size
inference_batch_size = Histogram(
    'nexus_inference_batch_size',
    'Batch size for inference requests',
    ['model_name'],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
)


# ============================================================================
# TRANSACTION METRICS
# ============================================================================

# Transaction processing
transactions_processed_total = Counter(
    'nexus_transactions_processed_total',
    'Total transactions processed',
    ['source', 'type']
)

suspicious_transactions_total = Counter(
    'nexus_suspicious_transactions_total',
    'Total suspicious transactions detected',
    ['typology', 'risk_level']
)

suspicious_by_country = Counter(
    'nexus_suspicious_by_country_total',
    'Suspicious transactions by country',
    ['country']
)

detections_by_typology = Counter(
    'nexus_detections_by_typology_total',
    'Detections grouped by AML typology',
    ['typology']
)

# Transaction amounts
transaction_amount = Histogram(
    'nexus_transaction_amount_dollars',
    'Transaction amounts in USD',
    ['transaction_type'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
)


# ============================================================================
# DATA QUALITY METRICS
# ============================================================================

# Data drift
data_drift_score = Gauge(
    'nexus_data_drift_score',
    'Data drift score (0=no drift, 1=complete drift)',
    ['feature_set']
)

missing_features = Counter(
    'nexus_missing_features_total',
    'Count of missing feature values',
    ['feature_name']
)

feature_out_of_range = Counter(
    'nexus_feature_out_of_range_total',
    'Features with values outside expected range',
    ['feature_name']
)

# Feature importance (live)
feature_importance = Gauge(
    'nexus_feature_importance',
    'Current feature importance scores',
    ['feature_name', 'model_name']
)


# ============================================================================
# ALERT METRICS
# ============================================================================

alerts_generated_total = Counter(
    'nexus_alerts_generated_total',
    'Total alerts generated',
    ['priority', 'category']
)

sars_generated_total = Counter(
    'nexus_sars_generated_total',
    'Total SARs generated',
    ['typology']
)

alert_processing_duration = Histogram(
    'nexus_alert_processing_duration_seconds',
    'Time to process and consolidate alerts',
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)


# ============================================================================
# API METRICS
# ============================================================================

api_requests_total = Counter(
    'nexus_api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status_code']
)

api_errors_total = Counter(
    'nexus_api_errors_total',
    'Total API errors',
    ['endpoint', 'error_type']
)

api_request_duration = Histogram(
    'nexus_api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)


# ============================================================================
# KAFKA/STREAMING METRICS
# ============================================================================

kafka_consumer_lag = Gauge(
    'nexus_kafka_consumer_lag',
    'Kafka consumer lag',
    ['topic', 'partition', 'consumer_group']
)

kafka_messages_consumed = Counter(
    'nexus_kafka_messages_consumed_total',
    'Total Kafka messages consumed',
    ['topic']
)

streaming_processing_rate = Gauge(
    'nexus_streaming_processing_rate',
    'Messages processed per second',
    ['pipeline']
)


# ============================================================================
# SYSTEM METRICS
# ============================================================================

memory_usage_bytes = Gauge(
    'nexus_memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

memory_total_bytes = Gauge(
    'nexus_memory_total_bytes',
    'Total available memory in bytes'
)

cpu_usage_percent = Gauge(
    'nexus_cpu_usage_percent',
    'CPU usage percentage',
    ['component']
)

gpu_utilization_percent = Gauge(
    'nexus_gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)


# ============================================================================
# ML OPS METRICS
# ============================================================================

model_last_training_timestamp = Gauge(
    'nexus_model_last_training_timestamp',
    'Timestamp of last model training',
    ['model_name']
)

training_job_status = Gauge(
    'nexus_training_job_status',
    'Status of training jobs (1=running, 0=stopped)',
    ['job_id', 'status']
)

model_version = Info(
    'nexus_model_version',
    'Current model version information'
)

suspicious_transactions_baseline = Gauge(
    'nexus_suspicious_transactions_baseline',
    'Baseline rate for suspicious transactions'
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

class MetricsExporter:
    """Central class for managing all NEXUS AI metrics"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        logger.info(f"Initializing MetricsExporter on port {port}")
    
    def start(self):
        """Start Prometheus metrics server"""
        start_http_server(self.port)
        logger.info(f"✅ Prometheus metrics server started on port {self.port}")
        logger.info(f"   Metrics available at: http://localhost:{self.port}/metrics")
    
    def record_inference(self, model_name: str, duration: float, success: bool):
        """Record inference metrics"""
        inference_duration.labels(model_name=model_name, endpoint='/predict').observe(duration)
        status = 'success' if success else 'error'
        inference_requests_total.labels(model_name=model_name, status=status).inc()
    
    def record_transaction(self, transaction_type: str, amount: float, is_suspicious: bool,
                          typology: str = None, country: str = None):
        """Record transaction metrics"""
        transactions_processed_total.labels(source='api', type=transaction_type).inc()
        transaction_amount.labels(transaction_type=transaction_type).observe(amount)
        
        if is_suspicious:
            risk_level = 'high' if amount > 50000 else 'medium' if amount > 10000 else 'low'
            suspicious_transactions_total.labels(
                typology=typology or 'unknown',
                risk_level=risk_level
            ).inc()
            
            if country:
                suspicious_by_country.labels(country=country).inc()
            
            if typology:
                detections_by_typology.labels(typology=typology).inc()
    
    def update_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Update model performance metrics"""
        model_accuracy.labels(model_name=model_name, environment='production').set(
            metrics.get('accuracy', 0)
        )
        model_precision.labels(model_name=model_name).set(metrics.get('precision', 0))
        model_recall.labels(model_name=model_name).set(metrics.get('recall', 0))
        model_f1_score.labels(model_name=model_name).set(metrics.get('f1', 0))
        model_roc_auc.labels(model_name=model_name).set(metrics.get('roc_auc', 0))
        
        fpr = metrics.get('false_positive_rate', 0)
        fnr = metrics.get('false_negative_rate', 0)
        false_positive_rate.labels(model_name=model_name).set(fpr)
        false_negative_rate.labels(model_name=model_name).set(fnr)
    
    def record_alert(self, priority: str, category: str):
        """Record alert generation"""
        alerts_generated_total.labels(priority=priority, category=category).inc()
    
    def record_sar(self, typology: str):
        """Record SAR generation"""
        sars_generated_total.labels(typology=typology).inc()
    
    def update_data_quality(self, drift_score: float, missing_count: Dict[str, int]):
        """Update data quality metrics"""
        data_drift_score.labels(feature_set='all').set(drift_score)
        
        for feature, count in missing_count.items():
            if count > 0:
                missing_features.labels(feature_name=feature).inc(count)


# ============================================================================
# DEMO/TESTING
# ============================================================================

def simulate_metrics(exporter: MetricsExporter, duration: int = 300):
    """Simulate realistic metrics for testing"""
    logger.info(f"Simulating metrics for {duration} seconds...")
    
    start_time = time.time()
    
    # Initialize baseline
    suspicious_transactions_baseline.set(0.05)
    
    while time.time() - start_time < duration:
        # Simulate inference
        inference_time = random.uniform(0.005, 0.15)
        exporter.record_inference('ensemble', inference_time, success=random.random() > 0.02)
        
        # Simulate transactions
        is_suspicious = random.random() < 0.05
        amount = random.lognormal(9, 1.5) if not is_suspicious else random.uniform(9000, 9900)
        typology = random.choice(['structuring', 'layering', 'smurfing']) if is_suspicious else None
        country = random.choice(['US', 'GB', 'CH', 'SG', 'BR'])
        
        exporter.record_transaction(
            transaction_type='wire',
            amount=amount,
            is_suspicious=is_suspicious,
            typology=typology,
            country=country
        )
        
        # Update model performance periodically
        if int(time.time()) % 30 == 0:
            exporter.update_model_performance('ensemble', {
                'accuracy': random.uniform(0.88, 0.95),
                'precision': random.uniform(0.85, 0.93),
                'recall': random.uniform(0.80, 0.92),
                'f1': random.uniform(0.82, 0.92),
                'roc_auc': random.uniform(0.90, 0.97),
                'false_positive_rate': random.uniform(0.03, 0.12),
                'false_negative_rate': random.uniform(0.05, 0.15)
            })
        
        # Simulate data drift
        if int(time.time()) % 60 == 0:
            exporter.update_data_quality(
                drift_score=random.uniform(0.0, 0.2),
                missing_count={'velocity_7d': random.randint(0, 3)}
            )
        
        time.sleep(random.uniform(0.01, 0.1))
    
    logger.info("Simulation complete")


if __name__ == "__main__":
    # Start metrics exporter
    exporter = MetricsExporter(port=9090)
    exporter.start()
    
    print("\n" + "="*70)
    print("🎯 NEXUS AI - Prometheus Metrics Exporter")
    print("="*70)
    print(f"\n📊 Metrics endpoint: http://localhost:9090/metrics")
    print(f"📈 Grafana dashboards: See monitoring/grafana/dashboards/")
    print(f"⚠️  Alert rules: See monitoring/alerts/nexus_alerts.yml")
    print("\n🔄 Starting metric simulation...")
    print("="*70 + "\n")
    
    # Run simulation
    try:
        simulate_metrics(exporter, duration=3600)  # Run for 1 hour
    except KeyboardInterrupt:
        print("\n\n✅ Metrics exporter stopped")

