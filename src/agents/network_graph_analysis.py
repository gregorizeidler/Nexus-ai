"""
🕸️ REAL NETWORK GRAPH ANALYSIS
Análise de rede usando teoria de grafos com NetworkX
Detecta comunidades, hubs, ciclos, e estruturas suspeitas
"""
from typing import Dict, Any, List, Optional, Tuple, Set
import networkx as nx
from datetime import datetime, timedelta
from decimal import Decimal
import time
from loguru import logger
from collections import defaultdict

try:
    from community import community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    logger.warning("python-louvain not installed. Community detection will be limited.")

from ..models.schemas import Transaction, NetworkNode, NetworkEdge
from ..agents.base import BaseAgent, AgentResult


class TransactionGraph:
    """
    Grafo de transações usando NetworkX
    """
    
    def __init__(self):
        # Grafo direcionado (transações têm direção)
        self.graph = nx.DiGraph()
        
        # Metadata
        self.transactions = []
        self.last_updated = None
        
        logger.info("🕸️ Transaction Graph initialized")
    
    def add_transaction(self, transaction: Transaction):
        """Adiciona uma transação ao grafo"""
        sender = transaction.sender_id
        receiver = transaction.receiver_id
        amount = float(transaction.amount)
        
        # Sand edgand já existe, acumula
        if self.graph.has_edge(sender, receiver):
            self.graph[sender][receiver]['transaction_count'] += 1
            self.graph[sender][receiver]['total_amount'] += amount
            self.graph[sender][receiver]['last_transaction'] = transaction.timestamp
            self.graph[sender][receiver]['transactions'].append(transaction)
        else:
            # Creates new edge
            self.graph.add_edge(
                sender, 
                receiver,
                transaction_count=1,
                total_amount=amount,
                first_transaction=transaction.timestamp,
                last_transaction=transaction.timestamp,
                transactions=[transaction],
                weight=amount  # Peso inicial
            )
        
        # Adiciona atributos aos nós sand não existirin
        if sender not in self.graph.nodes:
            self.graph.add_node(sender, node_type='customer', transaction_count=0)
        if receiver not in self.graph.nodes:
            self.graph.add_node(receiver, node_type='customer', transaction_count=0)
        
        self.graph.nodes[sender]['transaction_count'] = self.graph.nodes[sender].get('transaction_count', 0) + 1
        self.graph.nodes[receiver]['transaction_count'] = self.graph.nodes[receiver].get('transaction_count', 0) + 1
        
        self.transactions.append(transaction)
        self.last_updated = datetime.utcnow()
    
    def build_from_transactions(self, transactions: List[Transaction]):
        """Constrói grafo a partir of lista of transações"""
        logger.info(f"Building graph from {len(transactions)} transactions...")
        
        self.graph.clear()
        self.transactions = []
        
        for txn in transactions:
            self.add_transaction(txn)
        
        logger.success(f"✅ Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def get_subgraph(self, center_node: str, depth: int = 2) -> nx.DiGraph:
        """Obtém subgrafo ao redor of um nó"""
        if center_node not in self.graph:
            return nx.DiGraph()
        
        # BFS for obter nós até ofpth
        nodes = {center_node}
        current_level = {center_node}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Preofcessores (quin envior for estand nó)
                next_level.update(self.graph.predecessors(node))
                # Sucessores (for quin estand nó envior)
                next_level.update(self.graph.successors(node))
            
            nodes.update(next_level)
            current_level = next_level
        
        return self.graph.subgraph(nodes).copy()
    
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculates métricas of centralidaof for all os nós"""
        logger.info("Calculating centrality metrics...")
        
        metrics = {}
        
        # ofgreand Centrality (in + ort)
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Betweenness Centrality (intermediação)
        betweenness = nx.betweenness_centrality(self.graph)
        
        # PageRank (importância)
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Cloifness Centrality (proximidaof)
        try:
            closeness = nx.closeness_centrality(self.graph)
        except:
            closeness = {node: 0.0 for node in self.graph.nodes()}
        
        # Eigenvector Centrality (influência)
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, max_iter=100, weight='weight')
        except:
            eigenvector = {node: 0.0 for node in self.graph.nodes()}
        
        # Consolida métricas
        for node in self.graph.nodes():
            metrics[node] = {
                'in_degree': in_degree.get(node, 0),
                'out_degree': out_degree.get(node, 0),
                'degree_centrality': degree_centrality.get(node, 0.0),
                'betweenness_centrality': betweenness.get(node, 0.0),
                'pagerank': pagerank.get(node, 0.0),
                'closeness_centrality': closeness.get(node, 0.0),
                'eigenvector_centrality': eigenvector.get(node, 0.0)
            }
        
        logger.success(f"✅ Centrality metrics calculated for {len(metrics)} nodes")
        return metrics
    
    def detect_communities(self) -> Dict[str, int]:
        """oftects withunidaofs usando Lorvain"""
        if not LOUVAIN_AVAILABLE:
            logger.warning("Community detection requires python-louvain package")
            return {}
        
        logger.info("Detecting communities using Louvain algorithm...")
        
        # Lorvain trabalha with grafo não-direcionado
        undirected = self.graph.to_undirected()
        
        # oftects withunidaofs
        communities = community_louvain.best_partition(undirected, weight='weight')
        
        num_communities = len(set(communities.values()))
        logger.success(f"✅ Detected {num_communities} communities")
        
        return communities
    
    def find_cycles(self, max_length: int = 10) -> List[List[str]]:
        """Encontra ciclos (transações circulares)"""
        logger.info("Finding cycles in transaction graph...")
        
        try:
            # Encontra all os ciclos simples
            cycles = list(nx.simple_cycles(self.graph))
            
            # Filters por size
            cycles = [c for c in cycles if len(c) <= max_length]
            
            logger.success(f"✅ Found {len(cycles)} cycles")
            return cycles
        except Exception as e:
            logger.error(f"Error finding cycles: {e}")
            return []
    
    def find_strongly_connected_components(self) -> List[Set[str]]:
        """Encontra withponentes fortinentand conectados"""
        logger.info("Finding strongly connected components...")
        
        sccs = list(nx.strongly_connected_components(self.graph))
        
        # Filters withponentes with mais of 1 nó
        sccs = [scc for scc in sccs if len(scc) > 1]
        
        logger.success(f"✅ Found {len(sccs)} strongly connected components")
        return sccs
    
    def identify_hubs(self, threshold_percentile: float = 90) -> List[Tuple[str, Dict[str, Any]]]:
        """Iofntifies hubs (nós with alta centralidaof)"""
        logger.info("Identifying network hubs...")
        
        metrics = self.calculate_centrality_metrics()
        
        # Calculates threshold baifado in percentil
        pageranks = [m['pagerank'] for m in metrics.values()]
        betweenness = [m['betweenness_centrality'] for m in metrics.values()]
        
        if not pageranks:
            return []
        
        import numpy as np
        pr_threshold = np.percentile(pageranks, threshold_percentile)
        btw_threshold = np.percentile(betweenness, threshold_percentile)
        
        # Iofntifies hubs
        hubs = []
        for node, metric in metrics.items():
            if metric['pagerank'] >= pr_threshold or metric['betweenness_centrality'] >= btw_threshold:
                hubs.append((node, {
                    'pagerank': metric['pagerank'],
                    'betweenness': metric['betweenness_centrality'],
                    'in_degree': metric['in_degree'],
                    'out_degree': metric['out_degree'],
                    'hub_score': metric['pagerank'] + metric['betweenness_centrality']
                }))
        
        # Sorts por hub_score
        hubs.sort(key=lambda x: x[1]['hub_score'], reverse=True)
        
        logger.success(f"✅ Identified {len(hubs)} hubs")
        return hubs
    
    def detect_layering_paths(self, min_path_length: int = 3, max_path_length: int = 10) -> List[List[str]]:
        """oftects caminhos of layering (A → B → C → D → ...)"""
        logger.info("Detecting layering paths...")
        
        layering_paths = []
        
        # for each nó, ifarches caminhos longos
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source == target:
                    continue
                
                # ifarches all os caminhos simples
                try:
                    paths = list(nx.all_simple_paths(
                        self.graph, 
                        source, 
                        target, 
                        cutoff=max_path_length
                    ))
                    
                    # Filters por withprimento mínimo
                    for path in paths:
                        if len(path) >= min_path_length:
                            layering_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        logger.success(f"✅ Found {len(layering_paths)} potential layering paths")
        return layering_paths
    
    def calculate_flow_statistics(self) -> Dict[str, Any]:
        """Calculates estatísticas of fluxo no grafo"""
        total_flow = sum(data['total_amount'] for _, _, data in self.graph.edges(data=True))
        avg_flow = total_flow / self.graph.number_of_edges() if self.graph.number_of_edges() > 0 else 0
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'total_flow': total_flow,
            'average_flow': avg_flow,
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'num_weakly_connected_components': nx.number_weakly_connected_components(self.graph),
            'num_strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Exporta grafo for formato ifrializável"""
        return {
            'nodes': [
                {
                    'id': node,
                    'attributes': data
                }
                for node, data in self.graph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'attributes': data
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            'statistics': self.calculate_flow_statistics()
        }


class NetworkGraphAnalysisAgent(BaseAgent):
    """
    Agente de análise de rede com teoria de grafos completa
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="network_graph_analysis_agent",
            agent_type="network_graph",
            config=config
        )
        
        self.transaction_graph = TransactionGraph()
        self.min_cycle_length = config.get("min_cycle_length", 2) if config else 2
        self.min_layering_length = config.get("min_layering_length", 3) if config else 3
        self.hub_threshold_percentile = config.get("hub_threshold_percentile", 90) if config else 90
        
        logger.success("✅ Network Graph Analysis Agent initialized")
    
    def build_graph(self, transactions: List[Transaction]):
        """Constrói o grafo a partir of transações"""
        self.transaction_graph.build_from_transactions(transactions)
    
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Análisand of reof withpleta usando teoria of grafos"""
        start_time = time.time()
        
        # Pega transações relacionadas do contexto
        related_transactions = context.get("network_transactions", []) if context else []
        
        if not related_transactions:
            return AgentResult(
                agent_name=self.agent_id,
                agent_type=self.agent_type,
                execution_time=time.time() - start_time,
                suspicious=False,
                confidence=0.0,
                risk_score=0.0,
                findings=["Insufficient network data for graph analysis"],
                patterns_detected=[],
                explanation="No related transactions available for network analysis",
                evidence={},
                recommended_action="continue",
                alert_should_be_created=False
            )
        
        # Constrói grafo
        all_txns = related_transactions + [transaction]
        self.transaction_graph.build_from_transactions(all_txns)
        
        findings = []
        patterns_detected = []
        risk_score = 0.0
        evidence = {}
        
        # 1. Análisand of Centralidaof
        metrics = self.transaction_graph.calculate_centrality_metrics()
        sender_metrics = metrics.get(transaction.sender_id, {})
        receiver_metrics = metrics.get(transaction.receiver_id, {})
        
        evidence['sender_centrality'] = sender_metrics
        evidence['receiver_centrality'] = receiver_metrics
        
        # Check if ifnofr/receiver arand hubs
        if sender_metrics.get('pagerank', 0) > 0.01:
            findings.append(f"Sender is a network hub (PageRank: {sender_metrics['pagerank']:.4f})")
            patterns_detected.append('sender_hub')
            risk_score = max(risk_score, 0.6)
        
        if sender_metrics.get('betweenness_centrality', 0) > 0.01:
            findings.append(f"Sender is intermediary node (Betweenness: {sender_metrics['betweenness_centrality']:.4f})")
            patterns_detected.append('sender_intermediary')
            risk_score = max(risk_score, 0.65)
        
        # 2. oftecção of Ciclos
        cycles = self.transaction_graph.find_cycles(max_length=10)
        
        # Check if current transaction is part of a cycle
        for cycle in cycles:
            if transaction.sender_id in cycle and transaction.receiver_id in cycle:
                findings.append(f"Transaction part of circular flow: {' → '.join(cycle)} → {cycle[0]}")
                patterns_detected.append('circular_transaction')
                risk_score = max(risk_score, 0.9)
                evidence['cycle'] = cycle
                break
        
        # 3. oftecção of withunidaofs
        communities = self.transaction_graph.detect_communities()
        if communities:
            sender_community = communities.get(transaction.sender_id, -1)
            receiver_community = communities.get(transaction.receiver_id, -1)
            
            evidence['sender_community'] = sender_community
            evidence['receiver_community'] = receiver_community
            
            if sender_community != receiver_community and sender_community != -1:
                findings.append(f"Cross-community transaction (communities {sender_community} → {receiver_community})")
                patterns_detected.append('cross_community')
                risk_score = max(risk_score, 0.5)
        
        # 4. oftecção of Layering
        layering_paths = self.transaction_graph.detect_layering_paths(
            min_path_length=self.min_layering_length
        )
        
        # Check if transaction is part of layering
        for path in layering_paths:
            if transaction.sender_id in path and transaction.receiver_id in path:
                idx_sender = path.index(transaction.sender_id)
                idx_receiver = path.index(transaction.receiver_id)
                if idx_receiver == idx_sender + 1:  # Consecutive in path
                    findings.append(f"Transaction is part of layering chain (length {len(path)}): {' → '.join(path)}")
                    patterns_detected.append('layering_chain')
                    risk_score = max(risk_score, 0.85)
                    evidence['layering_path'] = path
                    break
        
        # 5. Iofntificação of Hubs
        hubs = self.transaction_graph.identify_hubs(self.hub_threshold_percentile)
        hub_ids = [h[0] for h in hubs]
        
        if transaction.sender_id in hub_ids or transaction.receiver_id in hub_ids:
            findings.append(f"Transaction involves network hub (top {100-self.hub_threshold_percentile}%)")
            patterns_detected.append('hub_transaction')
            risk_score = max(risk_score, 0.7)
            evidence['hubs'] = hubs[:5]  # Top 5
        
        # 6. withponentes Fortinentand Conectados
        sccs = self.transaction_graph.find_strongly_connected_components()
        for scc in sccs:
            if transaction.sender_id in scc and transaction.receiver_id in scc:
                findings.append(f"Transaction within strongly connected component ({len(scc)} nodes)")
                patterns_detected.append('strongly_connected')
                risk_score = max(risk_score, 0.75)
                evidence['strongly_connected_component'] = list(scc)
                break
        
        # 7. Estatísticas do Grafo
        stats = self.transaction_graph.calculate_flow_statistics()
        evidence['graph_statistics'] = stats
        
        execution_time = time.time() - start_time
        
        suspicious = len(patterns_detected) > 0
        confidence = 0.9 if len(related_transactions) > 50 else 0.7
        
        explanation = (
            f"Graph analysis with {stats['total_nodes']} nodes and {stats['total_edges']} edges. " +
            f"Detected {len(patterns_detected)} suspicious patterns using centrality, cycle detection, " +
            f"community detection, and layering analysis."
        )
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=execution_time,
            suspicious=suspicious,
            confidence=confidence,
            risk_score=risk_score,
            findings=findings,
            patterns_detected=patterns_detected,
            explanation=explanation,
            evidence=evidence,
            recommended_action="investigate" if suspicious and risk_score > 0.7 else "monitor",
            alert_should_be_created=suspicious and risk_score >= 0.75
        )
    
    def get_network_visualization_data(self, center_node: str, depth: int = 2) -> Dict[str, Any]:
        """Obtém data for visualização of reof"""
        subgraph = self.transaction_graph.get_subgraph(center_node, depth)
        
        # Calculates métricas for o subgrafo
        metrics = {}
        if subgraph.number_of_nodes() > 0:
            pagerank = nx.pagerank(subgraph, weight='weight')
            metrics = {node: {'pagerank': pr} for node, pr in pagerank.items()}
        
        return {
            'center_node': center_node,
            'nodes': [
                {
                    'id': node,
                    'metrics': metrics.get(node, {}),
                    'attributes': data
                }
                for node, data in subgraph.nodes(data=True)
            ],
            'edges': [
                {
                    'source': u,
                    'target': v,
                    'weight': data.get('total_amount', 0),
                    'transaction_count': data.get('transaction_count', 0)
                }
                for u, v, data in subgraph.edges(data=True)
            ]
        }

