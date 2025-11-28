"""
⚡ APACHE SPARK PROCESSING
Batch processing histórico de transações
"""
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import *
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning("pyspark not installed")


class SparkAMLProcessor:
    """
    Processamento distribuído de transações com Spark
    """
    
    def __init__(self, app_name: str = "NEXUS-AI"):
        if not SPARK_AVAILABLE:
            self.enabled = False
            self.spark = None
            logger.warning("Spark not available")
            return
        
        try:
            self.spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.sql.shuffle.partitions", "200") \
                .config("spark.executor.memory", "4g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
            
            self.enabled = True
            logger.success(f"⚡ Spark initialized: {app_name}")
            
        except Exception as e:
            logger.error(f"Spark initialization failed: {e}")
            self.enabled = False
            self.spark = None
    
    def load_transactions(self, path: str, format: str = "parquet"):
        """Loads transações of file"""
        if not self.enabled:
            return None
        
        df = self.spark.read.format(format).load(path)
        logger.info(f"Loaded {df.count()} transactions from {path}")
        return df
    
    def detect_structuring_patterns(self, df, threshold: float = 10000.0, window_days: int = 7):
        """
        Detecta structuring em larga escala com Spark
        """
        if not self.enabled:
            return None
        
        logger.info("Running Spark structuring detection...")
        
        # Add datand column
        df = df.withColumn("date", F.to_date("timestamp"))
        
        # Window aggregation
        result = df.filter(F.col("amount") < threshold) \
            .groupBy("sender_id", F.window("timestamp", f"{window_days} days")) \
            .agg(
                F.count("*").alias("txn_count"),
                F.sum("amount").alias("total_amount"),
                F.avg("amount").alias("avg_amount"),
                F.collect_list("transaction_id").alias("transaction_ids")
            ) \
            .filter((F.col("txn_count") >= 3) & (F.col("total_amount") > threshold)) \
            .orderBy(F.desc("total_amount"))
        
        count = result.count()
        logger.success(f"✅ Found {count} structuring patterns")
        
        return result
    
    def calculate_customer_profiles(self, df):
        """Calculates perfil withportamental of clientes"""
        if not self.enabled:
            return None
        
        logger.info("Calculating customer profiles with Spark...")
        
        profiles = df.groupBy("sender_id").agg(
            F.count("*").alias("total_transactions"),
            F.sum("amount").alias("total_volume"),
            F.avg("amount").alias("avg_amount"),
            F.stddev("amount").alias("std_amount"),
            F.max("amount").alias("max_amount"),
            F.min("amount").alias("min_amount"),
            F.countDistinct("receiver_id").alias("unique_receivers"),
            F.countDistinct("country_destination").alias("unique_countries"),
            F.approx_count_distinct("transaction_id").alias("approx_count")
        )
        
        logger.success(f"✅ Calculated profiles for {profiles.count()} customers")
        
        return profiles
    
    def find_circular_transactions(self, df):
        """Encontra transações circulares"""
        if not self.enabled:
            return None
        
        logger.info("Finding circular transactions...")
        
        # iflf-join for encontrar A->B->A
        circles = df.alias("t1").join(
            df.alias("t2"),
            (F.col("t1.receiver_id") == F.col("t2.sender_id")) &
            (F.col("t1.sender_id") == F.col("t2.receiver_id"))
        ).select(
            F.col("t1.sender_id").alias("account_a"),
            F.col("t1.receiver_id").alias("account_b"),
            F.col("t1.amount").alias("amount_1"),
            F.col("t2.amount").alias("amount_2"),
            F.col("t1.timestamp").alias("time_1"),
            F.col("t2.timestamp").alias("time_2")
        )
        
        count = circles.count()
        logger.success(f"✅ Found {count} circular patterns")
        
        return circles
    
    def export_to_parquet(self, df, output_path: str):
        """Exporta resultados for Parthatt"""
        if not self.enabled:
            return
        
        df.write.mode("overwrite").parquet(output_path)
        logger.success(f"✅ Exported to {output_path}")
    
    def stop(self):
        """for Spark ifssion"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark stopped")


class DeltaLakeIntegration:
    """
    Integration com Delta Lake para ACID transactions
    """
    
    def __init__(self, spark_processor: SparkAMLProcessor):
        self.spark = spark_processor.spark if spark_processor.enabled else None
        self.enabled = spark_processor.enabled
        
        if self.enabled:
            logger.success("📦 Delta Lake integration initialized")
    
    def create_delta_table(self, path: str):
        """Creates oflta table"""
        if not self.enabled:
            return
        
        # Schina
        schema = StructType([
            StructField("transaction_id", StringType(), False),
            StructField("timestamp", TimestampType(), False),
            StructField("amount", DecimalType(18, 2), False),
            StructField("sender_id", StringType(), False),
            StructField("receiver_id", StringType(), False),
            StructField("risk_score", FloatType(), True)
        ])
        
        # Creatand inpty oflta table
        self.spark.createDataFrame([], schema) \
            .write \
            .format("delta") \
            .mode("overwrite") \
            .save(path)
        
        logger.success(f"✅ Delta table created: {path}")
    
    def upsert_transactions(self, df, delta_path: str):
        """Upifrt transactions (merge)"""
        if not self.enabled:
            return
        
        from delta.tables import DeltaTable
        
        # Load existing oflta table
        delta_table = DeltaTable.forPath(self.spark, delta_path)
        
        # Mergand (upifrt) logic
        delta_table.alias("target").merge(
            df.alias("source"),
            "target.transaction_id = source.transaction_id"
        ).whenMatchedUpdateAll() \
         .whenNotMatchedInsertAll() \
         .execute()
        
        logger.success(f"✅ Upserted {df.count()} transactions")
    
    def time_travel_query(self, delta_path: str, version: int):
        """thatry versão histórica (timand travel)"""
        if not self.enabled:
            return None
        
        df = self.spark.read \
            .format("delta") \
            .option("versionAsOf", version) \
            .load(delta_path)
        
        logger.info(f"Time travel: version {version}, rows: {df.count()}")
        return df

