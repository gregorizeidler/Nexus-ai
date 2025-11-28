"""
🕸️ GRAPH NEURAL NETWORKS
GNN para análise avançada de redes transacionais
"""
from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch-geometric not installed")


class TransactionGNN(nn.Module):
    """
    Graph Neural Network para detecção de lavagem em redes
    """
    
    def __init__(self, num_features: int = 10, hidden_dim: int = 64, num_classes: int = 2):
        super(TransactionGNN, self).__init__()
        
        # 3-layer GCN
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
        logger.success("🕸️ Transaction GNN initialized")
    
    def forward(self, x, edge_index):
        """Forward pass"""
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Classification
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


class GATTransactionNetwork(nn.Module):
    """
    Graph Attention Network para análise de transações
    """
    
    def __init__(self, num_features: int = 10, hidden_dim: int = 64, num_heads: int = 4, num_classes: int = 2):
        super(GATTransactionNetwork, self).__init__()
        
        # GAT layers with attention
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=0.3)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        logger.success(f"🎯 GAT Network initialized (heads={num_heads})")
    
    def forward(self, x, edge_index):
        """Forward pass with attention"""
        # Layer 1 with multi-head attention
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Layer 2
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Classification
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


class GNNTrainer:
    """
    Trainer para modelos GNN
    """
    
    def __init__(self, model, device: str = 'cpu'):
        if not TORCH_GEOMETRIC_AVAILABLE:
            self.enabled = False
            logger.warning("GNN training not available")
            return
        
        self.enabled = True
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        logger.success(f"🎓 GNN Trainer initialized (device={device})")
    
    def train_epoch(self, data, train_mask):
        """Treina por uma época"""
        if not self.enabled:
            return 0.0
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward
        out = self.model(data.x, data.edge_index)
        
        # Loss apenas nos nós of treino
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data, test_mask):
        """Avalia mooflo"""
        if not self.enabled:
            return 0.0, 0.0
        
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # Accuracy
            correct = pred[test_mask] == data.y[test_mask]
            acc = int(correct.sum()) / int(test_mask.sum())
            
            # Loss
            loss = F.nll_loss(out[test_mask], data.y[test_mask])
        
        return acc, loss.item()
    
    def train_model(self, data, train_mask, test_mask, epochs: int = 200):
        """Treina mooflo withpleto"""
        if not self.enabled:
            return
        
        logger.info(f"Training GNN for {epochs} epochs...")
        
        best_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(data, train_mask)
            
            if epoch % 20 == 0:
                acc, test_loss = self.evaluate(data, test_mask)
                logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Test Acc={acc:.4f}")
                
                if acc > best_acc:
                    best_acc = acc
                    self.save_model(f"models/gnn_best.pt")
        
        logger.success(f"✅ Training complete! Best accuracy: {best_acc:.4f}")
    
    def save_model(self, path: str):
        """Saves mooflo"""
        if not self.enabled:
            return
        
        torch.save(self.model.state_dict(), path)
        logger.info(f"💾 Model saved: {path}")
    
    def load_model(self, path: str):
        """Loads mooflo"""
        if not self.enabled:
            return
        
        self.model.load_state_dict(torch.load(path))
        logger.info(f"📂 Model loaded: {path}")


def create_graph_data_from_transactions(transactions: List[Dict]) -> Optional[Any]:
    """
    Converte transações para formato PyTorch Geometric
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        return None
    
    # Extracts nós únicos (clientes)
    nodes = set()
    edges = []
    
    for txn in transactions:
        sender = txn['sender_id']
        receiver = txn['receiver_id']
        nodes.add(sender)
        nodes.add(receiver)
        edges.append((sender, receiver))
    
    # Mapeia IDs for indexs
    node_list = sorted(list(nodes))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Edgand inofx
    edge_index = torch.tensor([
        [node_to_idx[src] for src, _ in edges],
        [node_to_idx[dst] for _, dst in edges]
    ], dtype=torch.long)
    
    # Noof features (exinplo: grau, total transacionado, etc)
    num_nodes = len(node_list)
    x = torch.randn(num_nodes, 10)  # 10 features por nó
    
    # Labels (exinplo: 0=normal, 1=suspiciors)
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    # Creates Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    logger.success(f"✅ Graph created: {num_nodes} nodes, {len(edges)} edges")
    
    return data

