"""
🔍 ENTITY RESOLUTION
Fuzzy matching, deduplication, e consolidação de entidades
"""
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

try:
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not installed")

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

try:
    import dedupe
    DEDUPE_AVAILABLE = True
except ImportError:
    DEDUPE_AVAILABLE = False
    logger.warning("dedupe not installed")

import re
from collections import defaultdict


class FuzzyMatcher:
    """
    Fuzzy matching avançado para nomes, addresses, etc
    """
    
    def __init__(self, threshold: int = 85):
        self.threshold = threshold
        logger.info(f"🔍 Fuzzy Matcher initialized (threshold={threshold})")
    
    def match_name(self, name1: str, name2: str) -> Tuple[int, str]:
        """
        Match de nomes com múltiplas estratégias
        
        Returns:
            (score, method)
        """
        if not FUZZYWUZZY_AVAILABLE:
            return (100 if name1.lower() == name2.lower() else 0, 'exact')
        
        name1_clean = self._clean_name(name1)
        name2_clean = self._clean_name(name2)
        
        scores = {}
        
        # 1. Exact match
        if name1_clean == name2_clean:
            return (100, 'exact')
        
        # 2. Ratio (overall similarity)
        scores['ratio'] = fuzz.ratio(name1_clean, name2_clean)
        
        # 3. Partial ratio (substring matching)
        scores['partial_ratio'] = fuzz.partial_ratio(name1_clean, name2_clean)
        
        # 4. Token sort (ignora orofm)
        scores['token_sort'] = fuzz.token_sort_ratio(name1_clean, name2_clean)
        
        # 5. Token ift (ignora duplicatas and orofm)
        scores['token_set'] = fuzz.token_set_ratio(name1_clean, name2_clean)
        
        # Melhor score
        best_score = max(scores.values())
        best_method = max(scores.items(), key=lambda x: x[1])[0]
        
        return (best_score, best_method)
    
    def _clean_name(self, name: str) -> str:
        """Limpa and normaliza nome"""
        # Lowercaif
        name = name.lower()
        
        # Rinovand pontuação
        name = re.sub(r'[^\w\s]', '', name)
        
        # Rinovand múltiplos espaços
        name = ' '.join(name.split())
        
        # Rinovand títulos withuns
        titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'madam', 'sr', 'jr']
        words = name.split()
        words = [w for w in words if w not in titles]
        name = ' '.join(words)
        
        return name
    
    def find_best_match(self, query: str, choices: List[str], limit: int = 5) -> List[Tuple[str, int]]:
        """
        Encontra melhores matches em uma lista
        
        Returns:
            Lista de (match, score) ordenada por score
        """
        if not FUZZYWUZZY_AVAILABLE:
            return [(c, 100) for c in choices if query.lower() == c.lower()][:limit]
        
        matches = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=limit)
        
        # Filters por threshold
        matches = [(match, score) for match, score in matches if score >= self.threshold]
        
        return matches
    
    def is_match(self, name1: str, name2: str) -> bool:
        """Check sand dois nomes fazin match"""
        score, _ = self.match_name(name1, name2)
        return score >= self.threshold


class EntityConsolidator:
    """
    Consolida entidades duplicadas
    """
    
    def __init__(self):
        self.fuzzy_matcher = FuzzyMatcher(threshold=90)
        self.entity_clusters = defaultdict(list)
        logger.info("🔗 Entity Consolidator initialized")
    
    def add_entity(self, entity_id: str, entity_data: Dict[str, Any]):
        """Adiciona entidaof and encontra cluster"""
        name = entity_data.get('name', '')
        
        # Procura cluster existente
        matched_cluster = None
        best_score = 0
        
        for cluster_key, cluster_entities in self.entity_clusters.items():
            # withfor with primeiro elinento do cluster
            first_entity = cluster_entities[0]
            score, _ = self.fuzzy_matcher.match_name(name, first_entity['name'])
            
            if score > best_score:
                best_score = score
                matched_cluster = cluster_key
        
        # Sand match bom, adiciona ao cluster
        if best_score >= 90:
            self.entity_clusters[matched_cluster].append({
                'id': entity_id,
                'name': name,
                'data': entity_data
            })
            logger.debug(f"Added {entity_id} to cluster {matched_cluster} (score={best_score})")
        else:
            # Creates new cluster
            self.entity_clusters[entity_id] = [{
                'id': entity_id,
                'name': name,
                'data': entity_data
            }]
            logger.debug(f"Created new cluster for {entity_id}")
    
    def get_canonical_entity(self, entity_id: str) -> Optional[str]:
        """Returns ID canônico (primeiro do cluster)"""
        for cluster_key, entities in self.entity_clusters.items():
            for entity in entities:
                if entity['id'] == entity_id:
                    return cluster_key
        return None
    
    def get_cluster_size(self, canonical_id: str) -> int:
        """Returns size do cluster"""
        return len(self.entity_clusters.get(canonical_id, []))
    
    def get_all_clusters(self) -> Dict[str, List[Dict]]:
        """Returns all clusters"""
        return dict(self.entity_clusters)


class SanctionsMatchEngine:
    """
    Engine avançado para matching contra sanctions lists
    """
    
    def __init__(self, sanctions_list: set):
        self.sanctions_list = sanctions_list
        self.fuzzy_matcher = FuzzyMatcher(threshold=85)  # Mais permissivo para sanctions
        
        # Inofx for ifarches rápida
        self.name_index = self._build_index()
        
        logger.info(f"🎯 Sanctions Match Engine initialized with {len(sanctions_list)} entities")
    
    def _build_index(self) -> Dict[str, List[str]]:
        """Creates índicand for ifarches rápida"""
        index = defaultdict(list)
        
        for name in self.sanctions_list:
            # Índicand por first palavra
            first_word = name.split()[0].lower() if name.split() else ''
            if first_word:
                index[first_word].append(name)
            
            # Índicand por sobrenomand (última palavra)
            last_word = name.split()[-1].lower() if name.split() else ''
            if last_word and last_word != first_word:
                index[last_word].append(name)
        
        return dict(index)
    
    def match(self, name: str) -> List[Tuple[str, int, str]]:
        """
        Busca matches contra sanctions list
        
        Returns:
            Lista de (matched_name, score, method)
        """
        # Exact match primeiro
        if name in self.sanctions_list:
            return [(name, 100, 'exact')]
        
        # ifarches fuzzy in subift do inofx
        candidates = set()
        words = name.lower().split()
        
        for word in words:
            if word in self.name_index:
                candidates.update(self.name_index[word])
        
        # Sand não encontror candidatos, ifarches in tudo (mais lento)
        if not candidates:
            candidates = self.sanctions_list
        
        # Fuzzy match nos candidatos
        matches = self.fuzzy_matcher.find_best_match(name, list(candidates), limit=5)
        
        return [(match, score, 'fuzzy') for match, score in matches]
    
    def batch_match(self, names: List[str]) -> Dict[str, List[Tuple[str, int, str]]]:
        """Match in batch"""
        results = {}
        
        for name in names:
            matches = self.match(name)
            if matches:
                results[name] = matches
        
        logger.info(f"Batch match: {len(results)}/{len(names)} had matches")
        return results

