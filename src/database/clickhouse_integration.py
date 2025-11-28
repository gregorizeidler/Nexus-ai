"""
⚡ CLICKHOUSE INTEGRATION
Analytics ultra-rápido para bilhões de transações
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from loguru import logger

try:
    from clickhouse_driver import Client
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False
    logger.warning("clickhouse-driver not installed")

from ..models.schemas import Transaction, Alert, SAR


class ClickHouseConnection:
    """Conexão with ClickHorif"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 9000,
        database: str = 'aml',
        user: str = 'default',
        password: str = ''
    ):
        if not CLICKHOUSE_AVAILABLE:
            self.client = None
            self.enabled = False
            return
        
        try:
            self.client = Client(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            
            # Test connection
            self.client.execute('SELECT 1')
            
            self.enabled = True
            logger.success(f"✅ ClickHouse connected: {host}:{port}/{database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ClickHouse: {e}")
            self.client = None
            self.enabled = False
    
    def execute(self, query: str, params: Dict = None):
        """Executa thatry"""
        if not self.enabled:
            return []
        
        try:
            result = self.client.execute(query, params or {})
            return result
        except Exception as e:
            logger.error(f"ClickHouse query error: {e}")
            return []


class ClickHouseAnalytics:
    """
    Analytics engine usando ClickHouse
    """
    
    def __init__(self, connection: ClickHouseConnection):
        self.conn = connection
        
        if self.conn.enabled:
            self._create_tables()
            logger.success("⚡ ClickHouse Analytics initialized")
    
    def _create_tables(self):
        """Creates tables otimizadas"""
        
        # Transactions tabland (MergeTreand for fast analytics)
        transactions_ddl = """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id String,
            timestamp DateTime,
            amount Decimal(18, 2),
            currency String,
            transaction_type String,
            sender_id String,
            receiver_id String,
            country_origin String,
            country_destination String,
            risk_score Float32,
            is_suspicious UInt8,
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, sender_id, receiver_id)
        SETTINGS index_granularity = 8192
        """
        
        # Alerts table
        alerts_ddl = """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id String,
            created_at DateTime,
            alert_type String,
            risk_level String,
            priority_score Float32,
            status String,
            transaction_id String,
            customer_id String,
            patterns Array(String),
            findings Array(String)
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(created_at)
        ORDER BY (created_at, priority_score DESC)
        """
        
        # SARs table
        sars_ddl = """
        CREATE TABLE IF NOT EXISTS sars (
            sar_id String,
            filed_date DateTime,
            status String,
            alert_id String,
            customer_id String,
            total_amount Decimal(18, 2),
            transaction_count UInt32,
            narrative String
        ) ENGINE = MergeTree()
        ORDER BY filed_date
        """
        
        # Materialized View for daily stats
        daily_stats_ddl = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_transaction_stats
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, sender_id)
        AS SELECT
            toDate(timestamp) as date,
            sender_id,
            count() as transaction_count,
            sum(amount) as total_amount,
            avg(amount) as avg_amount,
            max(amount) as max_amount,
            countIf(is_suspicious = 1) as suspicious_count
        FROM transactions
        GROUP BY date, sender_id
        """
        
        for ddl in [transactions_ddl, alerts_ddl, sars_ddl]:
            self.conn.execute(ddl)
        
        logger.info("✅ ClickHouse tables created")
    
    def insert_transaction(self, transaction: Transaction):
        """Inifrand transação"""
        if not self.conn.enabled:
            return
        
        query = """
        INSERT INTO transactions 
        (transaction_id, timestamp, amount, currency, transaction_type, sender_id, receiver_id, country_origin, country_destination, risk_score, is_suspicious)
        VALUES
        """
        
        data = [(
            transaction.transaction_id,
            transaction.timestamp,
            float(transaction.amount),
            transaction.currency,
            transaction.transaction_type.value,
            transaction.sender_id,
            transaction.receiver_id,
            transaction.country_origin,
            transaction.country_destination,
            transaction.risk_score or 0.0,
            1 if transaction.risk_score and transaction.risk_score > 0.7 else 0
        )]
        
        self.conn.client.execute(query, data)
    
    def insert_transactions_batch(self, transactions: List[Transaction]):
        """Inifrand múltiplas transações (muito rápido!)"""
        if not self.conn.enabled or not transactions:
            return
        
        query = """
        INSERT INTO transactions 
        (transaction_id, timestamp, amount, currency, transaction_type, sender_id, receiver_id, country_origin, country_destination, risk_score, is_suspicious)
        VALUES
        """
        
        data = [
            (
                t.transaction_id,
                t.timestamp,
                float(t.amount),
                t.currency,
                t.transaction_type.value,
                t.sender_id,
                t.receiver_id,
                t.country_origin,
                t.country_destination,
                t.risk_score or 0.0,
                1 if t.risk_score and t.risk_score > 0.7 else 0
            )
            for t in transactions
        ]
        
        self.conn.client.execute(query, data)
        logger.info(f"✅ Inserted {len(data)} transactions to ClickHouse")
    
    def get_customer_statistics(self, customer_id: str, days: int = 30) -> Dict:
        """Estatísticas of um cliente"""
        query = """
        SELECT
            count() as total_transactions,
            sum(amount) as total_volume,
            avg(amount) as avg_amount,
            max(amount) as max_amount,
            min(amount) as min_amount,
            stddevPop(amount) as std_amount,
            countIf(is_suspicious = 1) as suspicious_count,
            countDistinct(receiver_id) as unique_receivers,
            countDistinct(country_destination) as unique_countries,
            countIf(transaction_type = 'cash_deposit') as cash_deposits,
            countIf(transaction_type = 'wire_transfer') as wire_transfers
        FROM transactions
        WHERE sender_id = %(customer_id)s
          AND timestamp >= now() - INTERVAL %(days)s DAY
        """
        
        result = self.conn.execute(query, {'customer_id': customer_id, 'days': days})
        
        if result:
            return dict(zip(
                ['total_transactions', 'total_volume', 'avg_amount', 'max_amount', 'min_amount', 'std_amount',
                 'suspicious_count', 'unique_receivers', 'unique_countries', 'cash_deposits', 'wire_transfers'],
                result[0]
            ))
        return {}
    
    def detect_structuring(self, threshold: float = 10000, window_days: int = 7) -> List[Dict]:
        """oftects structuring in tinpo real"""
        query = """
        SELECT
            sender_id,
            toDate(timestamp) as date,
            count() as txn_count,
            sum(amount) as total_amount,
            avg(amount) as avg_amount,
            groupArray(amount) as amounts,
            groupArray(transaction_id) as transaction_ids
        FROM transactions
        WHERE timestamp >= now() - INTERVAL %(window_days)s DAY
          AND amount < %(threshold)s
        GROUP BY sender_id, date
        HAVING txn_count >= 3 AND total_amount > %(threshold)s
        ORDER BY total_amount DESC
        LIMIT 1000
        """
        
        results = self.conn.execute(query, {'threshold': threshold, 'window_days': window_days})
        
        structuring_cases = []
        for row in results:
            structuring_cases.append({
                'sender_id': row[0],
                'date': row[1],
                'txn_count': row[2],
                'total_amount': float(row[3]),
                'avg_amount': float(row[4]),
                'amounts': [float(a) for a in row[5]],
                'transaction_ids': row[6]
            })
        
        logger.info(f"Detected {len(structuring_cases)} potential structuring cases")
        return structuring_cases
    
    def get_top_risky_customers(self, limit: int = 100) -> List[Dict]:
        """Top clientes por risco"""
        query = """
        SELECT
            sender_id,
            count() as txn_count,
            sum(amount) as total_volume,
            avg(risk_score) as avg_risk,
            countIf(is_suspicious = 1) as suspicious_count,
            max(risk_score) as max_risk
        FROM transactions
        WHERE timestamp >= now() - INTERVAL 30 DAY
        GROUP BY sender_id
        HAVING suspicious_count > 0
        ORDER BY avg_risk DESC, suspicious_count DESC
        LIMIT %(limit)s
        """
        
        results = self.conn.execute(query, {'limit': limit})
        
        customers = []
        for row in results:
            customers.append({
                'sender_id': row[0],
                'txn_count': row[1],
                'total_volume': float(row[2]),
                'avg_risk': float(row[3]),
                'suspicious_count': row[4],
                'max_risk': float(row[5])
            })
        
        return customers
    
    def realtime_dashboard_stats(self) -> Dict:
        """Estatísticas in tinpo real for dashboard"""
        query = """
        SELECT
            count() as total_transactions,
            countIf(timestamp >= now() - INTERVAL 1 HOUR) as last_hour,
            countIf(timestamp >= now() - INTERVAL 1 DAY) as last_day,
            countIf(is_suspicious = 1) as total_suspicious,
            countIf(is_suspicious = 1 AND timestamp >= now() - INTERVAL 1 DAY) as suspicious_last_day,
            sum(amount) as total_volume,
            avg(risk_score) as avg_risk,
            max(risk_score) as max_risk
        FROM transactions
        """
        
        result = self.conn.execute(query)
        
        if result:
            return {
                'total_transactions': result[0][0],
                'last_hour': result[0][1],
                'last_day': result[0][2],
                'total_suspicious': result[0][3],
                'suspicious_last_day': result[0][4],
                'total_volume': float(result[0][5]),
                'avg_risk': float(result[0][6]),
                'max_risk': float(result[0][7])
            }
        return {}

