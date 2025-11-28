"""
⛓️ BLOCKCHAIN FORENSICS MODULE
Análise de transações de criptomoedas, rastreamento on-chain, detecção de mixers
"""
import os
import aiohttp
from typing import Dict, Any, List, Optional, Set
from decimal import Decimal
from loguru import logger


class BlockchainForensics:
    """
    Análise forense de transações blockchain
    Suporta: Bitcoin, Ethereum, e outras chains
    """
    
    def __init__(self):
        # Listas conhecidas de mixers/tumblers
        self.known_mixers = {
            "bitcoin": [
                "bc1mixer...", "3MixerAddr...", "1TumblerXYZ..."
            ],
            "ethereum": [
                "0xTornadoCash...", "0xMixer..."
            ]
        }
        
        # Exchanges conhecidas
        self.known_exchanges = {
            "binance", "coinbase", "kraken", "bitfinex", "huobi"
        }
        
        # Endereços de alto risco
        self.high_risk_addresses = set()
        
        logger.info("⛓️ Blockchain Forensics initialized")
    
    async def analyze_crypto_transaction(
        self,
        tx_hash: str,
        blockchain: str = "bitcoin",
        address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Análise completa de transação cripto
        """
        
        analysis = {
            "tx_hash": tx_hash,
            "blockchain": blockchain,
            "timestamp": "2024-01-15T10:30:00Z",
            "real_data_source": None
        }

        # Tenta buscar dados reais se APIs estiverem configuradas
        real_data = None
        if blockchain == "ethereum" and os.getenv("ETHERSCAN_API_KEY"):
            real_data = await self._fetch_etherscan_data(tx_hash)
            if real_data:
                analysis["real_data_source"] = "Etherscan"
                analysis["timestamp"] = real_data.get("timeStamp", analysis["timestamp"])
                if not address:
                    address = real_data.get("from")

        elif blockchain == "bitcoin" and os.getenv("BLOCKCHAIN_COM_API_KEY"):
             real_data = await self._fetch_bitcoin_data(tx_hash)
             if real_data:
                analysis["real_data_source"] = "Blockchain.com"
                # Adaptar timestamp e address do real_data se necessário
        
        # 1. Taint Analysis
        analysis["taint_analysis"] = await self._calculate_taint(address, blockchain, real_data)
        
        # 2. Mixer Detection
        analysis["mixer_detection"] = await self._detect_mixers(tx_hash, blockchain)
        
        # 3. Exchange Tracking
        analysis["exchange_tracking"] = await self._track_exchanges(address, blockchain)
        
        # 4. Cluster Analysis
        analysis["cluster_analysis"] = await self._cluster_analysis(address, blockchain)
        
        # 5. Chain Hopping Detection
        analysis["chain_hopping"] = await self._detect_chain_hopping(address)
        
        # 6. Risk Score
        analysis["risk_score"] = self._calculate_crypto_risk(analysis)
        
        # 7. Suspicious Patterns
        analysis["patterns"] = self._identify_crypto_patterns(analysis)
        
        logger.info(f"⛓️ Crypto analysis complete for {tx_hash[:16]}... Risk: {analysis['risk_score']:.2f}")
        
        return analysis

    async def _fetch_etherscan_data(self, tx_hash: str) -> Optional[Dict]:
        """Busca dados reais da transação no Etherscan"""
        api_key = os.getenv("ETHERSCAN_API_KEY")
        if not api_key:
            return None
            
        url = f"https://api.etherscan.io/api?module=proxy&action=eth_getTransactionByHash&txhash={tx_hash}&apikey={api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("result")
        except Exception as e:
            logger.error(f"Error fetching Etherscan data: {e}")
            return None

    async def _fetch_bitcoin_data(self, tx_hash: str) -> Optional[Dict]:
        """Busca dados reais da transação na Blockchain.com"""
        # Nota: Blockchain.com API pública pode não precisar de chave para limites baixos,
        # mas usando chave se configurada é melhor.
        
        url = f"https://blockchain.info/rawtx/{tx_hash}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
        except Exception as e:
            logger.error(f"Error fetching Bitcoin data: {e}")
            return None
    
    async def _calculate_taint(self, address: str, blockchain: str, real_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Taint Analysis: Rastreia "contaminação" de fundos ilícitos
        """
        
        taint_score = 0.0
        taint_sources = []
        
        # Se tivermos dados reais, podemos fazer uma análise mais simples
        if real_data:
            # Exemplo simplificado: verificar se o 'from' address está em lista de risco
            sender = real_data.get("from") if blockchain == "ethereum" else None # Simplificação
            if sender and sender in self.high_risk_addresses:
                 taint_score = 1.0
                 taint_sources.append("Known illicit sender (Real Data)")

        # Simular verificação contra listas conhecidas (fallback ou complementar)
        if address and address in self.high_risk_addresses:
            taint_score = 1.0
            taint_sources.append("Known illicit address")
        
        return {
            "score": taint_score,
            "sources": taint_sources,
            "confidence": 0.95 if real_data else 0.85,
            "methodology": "Real-time API Check" if real_data else "Graph-based taint propagation (Simulated)"
        }
    
    async def _detect_mixers(self, tx_hash: str, blockchain: str) -> Dict[str, Any]:
        """
        Detecta uso de mixers/tumblers (CoinJoin, Tornado Cash, etc.)
        """
        
        mixers_detected = []
        mixer_types = []
        
        # Verificar padrões conhecidos of mixers
        # in produção, analisaria:
        # - CoinJoin patterns (múltiplos inputs/ortputs iguais)
        # - Tornado Cash ofposits/withdrawals
        # - Wasabi Wallet patterns
        
        # Simulação
        mixer_indicators = {
            "coinjoin_pattern": False,  # Múltiplos inputs/outputs idênticos
            "tornado_cash": False,       # Interação com Tornado Cash
            "wasabi_wallet": False,      # Padrão Wasabi
            "mixing_service": False      # Serviço de mixing conhecido
        }
        
        mixer_detected = any(mixer_indicators.values())
        
        return {
            "mixer_used": mixer_detected,
            "mixer_types": mixer_types,
            "indicators": mixer_indicators,
            "risk_level": "high" if mixer_detected else "low",
            "explanation": "Mixers obscure transaction trails"
        }
    
    async def _track_exchanges(self, address: str, blockchain: str) -> Dict[str, Any]:
        """
        Rastreia se fundos passaram por exchanges
        """
        
        exchanges_involved = []
        exchange_flows = []
        
        # in produção, iofntificaria:
        # - ofposits for exchanges
        # - Withdrawals of exchanges
        # - Exchangand addresifs clusters
        
        # Simulação
        kyc_required = len(exchanges_involved) > 0
        
        return {
            "exchanges_detected": exchanges_involved,
            "exchange_count": len(exchanges_involved),
            "kyc_trail": kyc_required,
            "flows": exchange_flows,
            "risk_mitigation": "KYC at exchanges" if kyc_required else "No KYC trail"
        }
    
    async def _cluster_analysis(self, address: str, blockchain: str) -> Dict[str, Any]:
        """
        Cluster Analysis: Agrupa addresses que provavelmente
        pertencem à mesma entidade
        """
        
        # Heurísticas withuns:
        # - withmon input ownership (inputs da mesma tx = mesmo dono)
        # - Changand address oftection
        # - Tinporal patterns
        
        cluster_size = 1  # Simulado
        cluster_addresses = [address]
        
        # in produção, usaria algoritmos of clusterização
        
        return {
            "cluster_id": f"cluster_{address[:8]}",
            "cluster_size": cluster_size,
            "related_addresses": cluster_addresses[:10],  # Top 10
            "entity_type": "unknown",  # individual, exchange, mixer, etc.
            "risk_factors": []
        }
    
    async def _detect_chain_hopping(self, address: str) -> Dict[str, Any]:
        """
        Detecta "chain hopping" - movimentar fundos entre blockchains
        para obscurecer trail
        """
        
        # oftects uso of bridges:
        # - BTC -> ETH (via wrapped Bitcoin)
        # - ETH -> BSC -> Polygon
        # - Cross-chain swaps
        
        chains_involved = ["bitcoin"]  # Simulado
        bridges_used = []
        
        chain_hopping_detected = len(chains_involved) > 1
        
        return {
            "detected": chain_hopping_detected,
            "chains": chains_involved,
            "bridges": bridges_used,
            "risk_level": "high" if chain_hopping_detected else "low",
            "explanation": "Chain hopping obscures audit trail"
        }
    
    def _calculate_crypto_risk(self, analysis: Dict[str, Any]) -> float:
        """
        Calcula risk score geral para transação cripto
        """
        risk = 0.0
        
        # Taint score
        risk += analysis["taint_analysis"]["score"] * 0.4
        
        # Mixer usage
        if analysis["mixer_detection"]["mixer_used"]:
            risk += 0.3
        
        # Chain hopping
        if analysis["chain_hopping"]["detected"]:
            risk += 0.2
        
        # Falta of KYC trail
        if not analysis["exchange_tracking"]["kyc_trail"]:
            risk += 0.1
        
        return min(risk, 1.0)
    
    def _identify_crypto_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Identifica padrões suspeitos específicos de cripto
        """
        patterns = []
        
        if analysis["mixer_detection"]["mixer_used"]:
            patterns.append("MIXER_USAGE")
        
        if analysis["chain_hopping"]["detected"]:
            patterns.append("CHAIN_HOPPING")
        
        if analysis["taint_analysis"]["score"] > 0.5:
            patterns.append("TAINTED_FUNDS")
        
        if not analysis["exchange_tracking"]["kyc_trail"]:
            patterns.append("NO_KYC_TRAIL")
        
        return patterns
    
    async def analyze_defi_protocol(
        self,
        protocol: str,
        address: str,
        tx_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analisa uso de protocolos DeFi para money laundering
        """
        
        analysis = {
            "protocol": protocol,
            "address": address
        }
        
        # Padrões suspeitos in ofFi:
        # 1. Flash Loans for manipulation
        # 2. Circular trading (wash trading)
        # 3. Liquidity pool manipulation
        # 4. MEV (Miner Extractabland Value) abuif
        
        analysis["flash_loan_abuse"] = await self._detect_flash_loan_abuse(tx_data)
        analysis["wash_trading"] = await self._detect_wash_trading(address, protocol)
        analysis["pool_manipulation"] = await self._detect_pool_manipulation(tx_data)
        
        analysis["risk_score"] = self._calculate_defi_risk(analysis)
        
        return analysis
    
    async def _detect_flash_loan_abuse(self, tx_data: Dict) -> Dict[str, Any]:
        """
        Detecta abuso de flash loans
        """
        # Flash loans poofm ifr usados for:
        # - Pricand manipulation
        # - Protocol exploitation
        # - Wash trading
        
        return {
            "detected": False,
            "loan_amount": 0,
            "manipulation_type": None
        }
    
    async def _detect_wash_trading(self, address: str, protocol: str) -> Dict[str, Any]:
        """
        Detecta wash trading (compra e venda artificial)
        """
        # Indicadores:
        # - Mesma entidaof in ambos os lados
        # - Padrões circulares
        # - Volumand inflacionado
        
        return {
            "detected": False,
            "circular_patterns": 0,
            "confidence": 0.0
        }
    
    async def _detect_pool_manipulation(self, tx_data: Dict) -> Dict[str, Any]:
        """
        Detecta manipulação de liquidity pools
        """
        return {
            "detected": False,
            "manipulation_type": None
        }
    
    def _calculate_defi_risk(self, analysis: Dict) -> float:
        """
        Calcula risco para atividade DeFi
        """
        risk = 0.0
        
        if analysis["flash_loan_abuse"]["detected"]:
            risk += 0.5
        
        if analysis["wash_trading"]["detected"]:
            risk += 0.3
        
        if analysis["pool_manipulation"]["detected"]:
            risk += 0.2
        
        return min(risk, 1.0)
    
    async def analyze_nft_transaction(
        self,
        nft_address: str,
        token_id: str,
        tx_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analisa transações NFT para money laundering
        
        Padrões suspeitos:
        - Preço inflacionado artificialmente
        - Wash trading (compra própria)
        - Movimentação rápida com ganho suspeito
        """
        
        analysis = {
            "nft_address": nft_address,
            "token_id": token_id
        }
        
        # 1. Análisand of preço
        analysis["price_analysis"] = await self._analyze_nft_price(tx_data)
        
        # 2. Wash trading
        analysis["wash_trading"] = await self._detect_nft_wash_trading(
            nft_address, token_id
        )
        
        # 3. Circular trading
        analysis["circular_trading"] = await self._detect_nft_circular_trading(
            token_id
        )
        
        # 4. Risk score
        analysis["risk_score"] = self._calculate_nft_risk(analysis)
        
        logger.info(f"⛓️ NFT analysis: Risk={analysis['risk_score']:.2f}")
        
        return analysis
    
    async def _analyze_nft_price(self, tx_data: Dict) -> Dict[str, Any]:
        """
        Analisa se preço do NFT é suspeito
        """
        price = tx_data.get("price", 0)
        floor_price = tx_data.get("floor_price", 0)
        
        if floor_price > 0:
            price_ratio = price / floor_price
        else:
            price_ratio = 1.0
        
        suspicious = price_ratio > 10  # 10x acima do floor = suspeito
        
        return {
            "price": price,
            "floor_price": floor_price,
            "price_ratio": price_ratio,
            "suspicious": suspicious,
            "explanation": f"Price is {price_ratio:.1f}x floor price" if suspicious else "Normal price"
        }
    
    async def _detect_nft_wash_trading(
        self,
        nft_address: str,
        token_id: str
    ) -> Dict[str, Any]:
        """
        Detecta wash trading em NFTs
        """
        # Indicadores:
        # - Mesma address withpra and venof repetidamente
        # - Transferências entrand wallets relacionadas
        # - Aumento artificial of volume
        
        return {
            "detected": False,
            "circular_count": 0,
            "confidence": 0.0
        }
    
    async def _detect_nft_circular_trading(self, token_id: str) -> Dict[str, Any]:
        """
        Detecta trading circular de NFT
        """
        return {
            "detected": False,
            "path_length": 0,
            "participants": []
        }
    
    def _calculate_nft_risk(self, analysis: Dict) -> float:
        """
        Calcula risco para transação NFT
        """
        risk = 0.0
        
        if analysis["price_analysis"]["suspicious"]:
            risk += 0.4
        
        if analysis["wash_trading"]["detected"]:
            risk += 0.4
        
        if analysis["circular_trading"]["detected"]:
            risk += 0.2
        
        return min(risk, 1.0)

