"""
🕸️ NEO4J INTEGRATION
Banco de dados de grafos persistente para análise de rede
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal
from loguru import logger

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j package not installed")

from ..models.schemas import Transaction, NetworkNode, NetworkEdge


class Neo4jConnection:
    """Conexão with Neo4j"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "amlpassword"):
        if not NEO4J_AVAILABLE:
            self.driver = None
            self.enabled = False
            return
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.enabled = True
            logger.success(f"✅ Neo4j connected: {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            self.enabled = False
    
    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Dict = None):
        """Executa thatry Cypher"""
        if not self.enabled:
            return []
        
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write(self, query: str, parameters: Dict = None):
        """Executa writand thatry"""
        if not self.enabled:
            return None
        
        with self.driver.session() as session:
            result = session.write_transaction(lambda tx: tx.run(query, parameters or {}))
            return result


class Neo4jTransactionGraph:
    """
    Grafo de transações persistente em Neo4j
    """
    
    def __init__(self, connection: Neo4jConnection):
        self.conn = connection
        
        if self.conn.enabled:
            self._create_indexes()
            logger.success("🕸️ Neo4j Transaction Graph initialized")
        else:
            logger.warning("Neo4j not available")
    
    def _create_indexes(self):
        """Creates indexs for performance"""
        queries = [
            "CREATE INDEX customer_id_idx IF NOT EXISTS FOR (c:Customer) ON (c.id)",
            "CREATE INDEX transaction_id_idx IF NOT EXISTS FOR ()-[t:SENT]-() ON (t.transaction_id)",
            "CREATE INDEX timestamp_idx IF NOT EXISTS FOR ()-[t:SENT]-() ON (t.timestamp)",
            "CREATE INDEX amount_idx IF NOT EXISTS FOR ()-[t:SENT]-() ON (t.amount)"
        ]
        
        for query in queries:
            try:
                self.conn.execute_write(query)
            except Exception as e:
                logger.debug(f"Index creation: {e}")
        
        logger.info("✅ Neo4j indexes created")
    
    def add_transaction(self, transaction: Transaction):
        """Adiciona transação ao grafo"""
        if not self.conn.enabled:
            return
        
        query = """
        MERGE (sender:Customer {id: $sender_id})
        ON CREATE SET 
            sender.created_at = datetime(),
            sender.transaction_count = 0
        ON MATCH SET
            sender.transaction_count = sender.transaction_count + 1,
            sender.last_activity = datetime()
        
        MERGE (receiver:Customer {id: $receiver_id})
        ON CREATE SET 
            receiver.created_at = datetime(),
            receiver.transaction_count = 0
        ON MATCH SET
            receiver.transaction_count = receiver.transaction_count + 1,
            receiver.last_activity = datetime()
        
        CREATE (sender)-[txn:SENT {
            transaction_id: $transaction_id,
            amount: $amount,
            currency: $currency,
            timestamp: datetime($timestamp),
            transaction_type: $transaction_type,
            country_origin: $country_origin,
            country_destination: $country_destination,
            risk_score: $risk_score
        }]->(receiver)
        
        RETURN txn
        """
        
        params = {
            'sender_id': transaction.sender_id,
            'receiver_id': transaction.receiver_id,
            'transaction_id': transaction.transaction_id,
            'amount': float(transaction.amount),
            'currency': transaction.currency,
            'timestamp': transaction.timestamp.isoformat(),
            'transaction_type': transaction.transaction_type.value,
            'country_origin': transaction.country_origin,
            'country_destination': transaction.country_destination,
            'risk_score': transaction.risk_score or 0.0
        }
        
        self.conn.execute_write(query, params)
        logger.debug(f"📝 Transaction added to Neo4j: {transaction.transaction_id}")
    
    def find_cycles(self, max_length: int = 10) -> List[List[str]]:
        """Encontra ciclos no grafo"""
        query = f"""
        MATCH path = (a:Customer)-[:SENT*2..{max_length}]->(a)
        RETURN [node in nodes(path) | node.id] as cycle,
               length(path) as cycle_length
        ORDER BY cycle_length DESC
        LIMIT 100
        """
        
        results = self.conn.execute_query(query)
        cycles = [r['cycle'] for r in results]
        
        logger.info(f"Found {len(cycles)} cycles in Neo4j")
        return cycles
    
    def find_layering_paths(self, min_length: int = 3, max_length: int = 10) -> List[Dict]:
        """Encontra caminhos of layering"""
        query = f"""
        MATCH path = (a:Customer)-[:SENT*{min_length}..{max_length}]->(b:Customer)
        WHERE a <> b
        WITH path, 
             [rel in relationships(path) | rel.amount] as amounts,
             [rel in relationships(path) | rel.timestamp] as timestamps
        RETURN [node in nodes(path) | node.id] as path_nodes,
               length(path) as path_length,
               reduce(total = 0.0, amt in amounts | total + amt) as total_amount,
               head(timestamps) as start_time,
               last(timestamps) as end_time
        ORDER BY path_length DESC, total_amount DESC
        LIMIT 100
        """
        
        results = self.conn.execute_query(query)
        
        logger.info(f"Found {len(results)} layering paths")
        return results
    
    def calculate_pagerank(self):
        """Calculates PageRank usando GDS"""
        queries = [
            # Project graph
            """
            CALL gds.graph.project(
                'transaction-graph',
                'Customer',
                'SENT',
                {
                    relationshipProperties: 'amount'
                }
            )
            """,
            
            # Run PageRank
            """
            CALL gds.pageRank.write('transaction-graph', {
                writeProperty: 'pagerank',
                relationshipWeightProperty: 'amount'
            })
            YIELD nodePropertiesWritten, ranIterations
            RETURN nodePropertiesWritten, ranIterations
            """
        ]
        
        try:
            for query in queries:
                self.conn.execute_write(query)
            logger.success("✅ PageRank calculated")
        except Exception as e:
            logger.error(f"PageRank calculation failed: {e}")
    
    def detect_communities(self):
        """oftects withunidaofs with Lorvain"""
        query = """
        CALL gds.louvain.write('transaction-graph', {
            writeProperty: 'community',
            relationshipWeightProperty: 'amount'
        })
        YIELD communityCount, modularity
        RETURN communityCount, modularity
        """
        
        try:
            result = self.conn.execute_write(query)
            logger.success(f"✅ Communities detected: {result}")
            return result
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            return None
    
    def get_customer_network(self, customer_id: str, depth: int = 2) -> Dict:
        """Obtém reof ao redor of um cliente"""
        query = f"""
        MATCH path = (center:Customer {{id: $customer_id}})-[:SENT*1..{depth}]-(other:Customer)
        WITH collect(DISTINCT center) + collect(DISTINCT other) as nodes,
             collect(DISTINCT relationships(path)) as rels
        UNWIND nodes as node
        UNWIND rels as rel_list
        UNWIND rel_list as rel
        RETURN DISTINCT 
            collect(DISTINCT {{
                id: node.id, 
                pagerank: node.pagerank,
                community: node.community,
                transaction_count: node.transaction_count
            }}) as nodes,
            collect(DISTINCT {{
                source: startNode(rel).id,
                target: endNode(rel).id,
                amount: rel.amount,
                timestamp: rel.timestamp
            }}) as edges
        """
        
        result = self.conn.execute_query(query, {'customer_id': customer_id})
        
        if result:
            return result[0]
        return {'nodes': [], 'edges': []}
    
    def get_high_risk_customers(self, limit: int = 100) -> List[Dict]:
        """Iofntifies clientes of alto risco baifado in reof"""
        query = """
        MATCH (c:Customer)
        WHERE c.pagerank IS NOT NULL
        WITH c, 
             c.pagerank as pr,
             c.transaction_count as txn_count
        ORDER BY pr DESC, txn_count DESC
        LIMIT $limit
        RETURN c.id as customer_id,
               pr as pagerank,
               txn_count as transactions,
               c.community as community
        """
        
        results = self.conn.execute_query(query, {'limit': limit})
        logger.info(f"Retrieved {len(results)} high-risk customers")
        return results
    
    def clear_graph(self):
        """Limpa todo o grafo (cuidado!)"""
        query = "MATCH (n) DETACH DELETE n"
        self.conn.execute_write(query)
        logger.warning("⚠️ Neo4j graph cleared")

