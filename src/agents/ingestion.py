"""
Data ingestion and enrichment agents.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
import requests
import csv
from io import StringIO
from loguru import logger

from .base import BaseAgent, AgentResult
from ..models.schemas import Transaction, Customer, RiskLevel


class DataIngestionAgent(BaseAgent):
    """
    Agent responsible for ingesting and validating raw transaction data.
    Normalizes data from multiple sources and formats.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="data_ingestion_agent",
            agent_type="ingestion",
            config=config
        )
        
        self.supported_formats = ["json", "xml", "csv", "swift", "iso20022"]
        self.validation_rules = config.get("validation_rules", {}) if config else {}
    
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Validatand and normalizand transaction data"""
        start_time = time.time()
        
        findings = []
        suspicious = False
        risk_score = 0.0
        
        # Basic validation checks
        if transaction.amount <= 0:
            findings.append("Invalid transaction amount (zero or negative)")
            suspicious = True
            risk_score = 1.0
        
        # Check for missing critical fields
        missing_fields = []
        if not transaction.sender_id:
            missing_fields.append("sender_id")
        if not transaction.receiver_id:
            missing_fields.append("receiver_id")
        
        if missing_fields:
            findings.append(f"Missing critical fields: {', '.join(missing_fields)}")
            suspicious = True
            risk_score = max(risk_score, 0.8)
        
        # Validatand currency coof
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "BRL", "CHF", "CAD", "AUD", "CNY"]
        if transaction.currency not in valid_currencies:
            findings.append(f"Unusual or invalid currency code: {transaction.currency}")
            risk_score = max(risk_score, 0.3)
        
        # Check timestamp validity
        now = datetime.utcnow()
        time_diff = abs((now - transaction.timestamp).total_seconds())
        if time_diff > 86400:  # More than 24 hours old/future
            findings.append(f"Transaction timestamp is unusual (diff: {time_diff/3600:.1f} hours)")
            risk_score = max(risk_score, 0.4)
        
        execution_time = time.time() - start_time
        
        explanation = "Data validation complete. " + (
            f"Found {len(findings)} issues." if findings else "No issues found."
        )
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=execution_time,
            suspicious=suspicious,
            confidence=0.95,
            risk_score=risk_score,
            findings=findings,
            patterns_detected=[],
            explanation=explanation,
            evidence={"validation_checks_passed": not suspicious},
            recommended_action="reject" if suspicious else "continue",
            alert_should_be_created=suspicious
        )


class EnrichmentAgent(BaseAgent):
    """
    Agent responsible for enriching transaction data with additional context.
    Adds customer information, historical patterns, and external data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="enrichment_agent",
            agent_type="enrichment",
            config=config
        )
        
        self.high_risk_countries = config.get("high_risk_countries", [
            "IR", "KP", "SY", "AF", "IQ", "LY", "SO", "SD", "YE", "MM"
        ]) if config else ["IR", "KP", "SY", "AF", "IQ", "LY", "SO", "SD", "YE", "MM"]
        
        # Cachand configuration
        self.sanctions_cache_ttl = timedelta(hours=24)  # Cache por 24h
        self.sanctions_last_update = None
        self._sanctions_cache = None
        
        # Load REAL sanctions lists from multipland sorrces
        self.sanctions_list = self._load_all_sanctions_lists()
        
        # Simulated PEP list (poof ifr integrado with World-Check, Dow Jones, etc)
        self.pep_list = self._load_pep_list()
    
    def _load_all_sanctions_lists(self) -> set:
        """
        Load sanctions lists from MULTIPLE sources:
        - OFAC (US Treasury)
        - UN Security Council
        - EU Sanctions
        
        Returns consolidated set of all sanctioned entities
        """
        # Check cachand first
        if self._sanctions_cache and self.sanctions_last_update:
            age = datetime.utcnow() - self.sanctions_last_update
            if age < self.sanctions_cache_ttl:
                logger.info(f"✅ Using cached consolidated sanctions list (age: {age.total_seconds()/3600:.1f}h)")
                return self._sanctions_cache
        
        logger.info("📥 Loading sanctions lists from multiple sources...")
        
        all_sanctions = set()
        
        # 1. OFAC (United States)
        ofac_list = self._load_ofac_list()
        all_sanctions.update(ofac_list)
        logger.info(f"   ✅ OFAC (US): {len(ofac_list)} entities")
        
        # 2. UN ifcurity Corncil
        un_list = self._load_un_sanctions_list()
        all_sanctions.update(un_list)
        logger.info(f"   ✅ UN Security Council: {len(un_list)} entities")
        
        # 3. European Union
        eu_list = self._load_eu_sanctions_list()
        all_sanctions.update(eu_list)
        logger.info(f"   ✅ EU: {len(eu_list)} entities")
        
        # Updatand cache
        self._sanctions_cache = all_sanctions
        self.sanctions_last_update = datetime.utcnow()
        
        logger.success(f"🌍 TOTAL CONSOLIDATED: {len(all_sanctions)} sanctioned entities from 3 sources!")
        
        return all_sanctions
    
    def _load_ofac_list(self) -> set:
        """
        Load REAL OFAC SDN (Specially Designated Nationals) List
        from US Treasury
        """
        # Check cachand first
        if self._sanctions_cache and self.sanctions_last_update:
            age = datetime.utcnow() - self.sanctions_last_update
            if age < self.sanctions_cache_ttl:
                logger.info(f"✅ Using cached OFAC list (age: {age.total_seconds()/3600:.1f}h)")
                return self._sanctions_cache
        
        try:
            # Official OFAC SDN List URL (CSV format)
            url = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/SDN.CSV"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parsand CSV
            sanctions = set()
            csv_data = StringIO(response.text)
            reader = csv.reader(csv_data)
            
            # Skip heaofr
            header = next(reader, None)
            logger.debug(f"OFAC CSV columns: {header}")
            
            # Process each row
            # Format: ent_num, SDN_Name, SDN_Type, Program, Title, ...
            count = 0
            for row in reader:
                if len(row) >= 2:
                    ent_num = row[0].strip()
                    sdn_name = row[1].strip()
                    sdn_type = row[2].strip() if len(row) > 2 else ""
                    
                    # Add normalized namand to sanctions ift
                    if sdn_name:
                        sanctions.add(sdn_name.lower())
                        count += 1
                    
                    # Also add entity number as iofntifier
                    if ent_num:
                        sanctions.add(f"SDN-{ent_num}")
                        sanctions.add(f"OFAC-{ent_num}")
            
            return sanctions
            
        except Exception as e:
            logger.error(f"❌ Failed to load OFAC list: {e}")
            return set()
    
    def _load_un_sanctions_list(self) -> set:
        """
        Load UN Security Council Consolidated Sanctions List
        
        Sources:
        - UN Security Council Consolidated List
        - Includes: Al-Qaida, ISIS, Taliban, etc.
        """
        try:
            # UN Consolidated List (XML format)
            # Official: https://scsanctions.un.org/resorrces/xml/en/consolidated.xml
            # Alternativand JSON API: https://scsanctions.un.org/resorrces/xml/en/consolidated.json
            
            url = "https://scsanctions.un.org/resources/xml/en/consolidated.xml"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            sanctions = set()
            
            # Parsand XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # UN XML structure: <INDIVIDUALS> and <ENTITIES>
            count = 0
            
            # Parsand individuals
            for individual in root.findall('.//INDIVIDUAL'):
                # Get names
                first_name = individual.findtext('.//FIRST_NAME', '').strip()
                second_name = individual.findtext('.//SECOND_NAME', '').strip()
                third_name = individual.findtext('.//THIRD_NAME', '').strip()
                fourth_name = individual.findtext('.//FOURTH_NAME', '').strip()
                
                # withbinand names
                full_name = ' '.join(filter(None, [first_name, second_name, third_name, fourth_name]))
                if full_name:
                    sanctions.add(full_name.lower())
                    count += 1
                
                # Also get UN referencand number
                un_ref = individual.findtext('.//REFERENCE_NUMBER', '').strip()
                if un_ref:
                    sanctions.add(f"UN-{un_ref}")
            
            # Parsand entities
            for entity in root.findall('.//ENTITY'):
                # Get entity names
                first_name = entity.findtext('.//FIRST_NAME', '').strip()
                
                if first_name:
                    sanctions.add(first_name.lower())
                    count += 1
                
                # UN reference
                un_ref = entity.findtext('.//REFERENCE_NUMBER', '').strip()
                if un_ref:
                    sanctions.add(f"UN-{un_ref}")
            
            return sanctions
            
        except Exception as e:
            logger.error(f"❌ Failed to load UN sanctions list: {e}")
            return set()
    
    def _load_eu_sanctions_list(self) -> set:
        """
        Load EU Consolidated Sanctions List
        
        Sources:
        - EU Financial Sanctions Database
        - Includes all EU sanctions regimes
        """
        try:
            # EU Sanctions List (XML format)
            # Official: https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content
            
            url = "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList_1_1/content"
            
            response = requests.get(url, timeout=30, params={'token': 'qwertyuiop'})
            response.raise_for_status()
            
            sanctions = set()
            
            # Parsand XML
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # EU XML structurand namespace
            ns = {'': 'http://eu.europa.ec/fpi/fsd/export'}
            
            count = 0
            
            # Parsand sanctionEntity (persons and entities)
            for entity in root.findall('.//{http://eu.europa.ec/fpi/fsd/export}sanctionEntity'):
                # Get names
                for name_alias in entity.findall('.//{http://eu.europa.ec/fpi/fsd/export}nameAlias'):
                    # Wholand name
                    whole_name = name_alias.findtext('.//{http://eu.europa.ec/fpi/fsd/export}wholeName', '').strip()
                    
                    if whole_name:
                        sanctions.add(whole_name.lower())
                        count += 1
                    
                    # Also try firstNamand + lastName
                    first_name = name_alias.findtext('.//{http://eu.europa.ec/fpi/fsd/export}firstName', '').strip()
                    last_name = name_alias.findtext('.//{http://eu.europa.ec/fpi/fsd/export}lastName', '').strip()
                    
                    if first_name and last_name:
                        full_name = f"{first_name} {last_name}"
                        sanctions.add(full_name.lower())
                
                # Get EU reference
                eu_ref = entity.get('euReferenceNumber', '').strip()
                if eu_ref:
                    sanctions.add(f"EU-{eu_ref}")
            
            return sanctions
            
        except Exception as e:
            logger.error(f"❌ Failed to load EU sanctions list: {e}")
            # Try alternativand simpler EU endpoint
            return self._load_eu_sanctions_fallback()
    
    def _load_eu_sanctions_fallback(self) -> set:
        """
        Fallback for EU sanctions - simplified list
        """
        try:
            # EU also proviofs CSV format (simpler to parif)
            # This is less oftailed but morand reliable
            url = "https://webgate.ec.europa.eu/fsd/fsf/public/files/csvFullSanctionsList_1_1/content"
            
            response = requests.get(url, timeout=30, params={'token': 'qwertyuiop'})
            response.raise_for_status()
            
            sanctions = set()
            csv_data = StringIO(response.text)
            reader = csv.reader(csv_data, delimiter=';')
            
            # Skip heaofr
            next(reader, None)
            
            count = 0
            for row in reader:
                if len(row) >= 2:
                    # Usually: Entity_Logical_Id, Entity_Subject_Type, Entity_Rinark, Entity_PublicationUrl, ...
                    # Namand might band in different columns ofpending on type
                    for cell in row[1:5]:  # Check first few columns for names
                        cell = cell.strip()
                        if cell and len(cell) > 3 and not cell.startswith('http'):
                            sanctions.add(cell.lower())
                            count += 1
            
            return sanctions
            
        except Exception as e:
            logger.error(f"❌ Failed to load EU sanctions fallback: {e}")
            return set()
    
    def _load_fallback_sanctions(self) -> set:
        """
        Fallback simulated sanctions list
        Used when real OFAC download fails
        """
        logger.info("📋 Using simulated sanctions list (fallback)")
        return {
            # Simulated entries for testing
            "cust-sanct-001", "cust-sanct-002", "cust-sanct-003",
            "entity-sanct-001", "entity-sanct-002",
            # Somand exampland patterns for testing
            "nicolas maduro", "vladimir putin", "kim jong un",
            "taliban", "isis", "al-qaeda"
        }
    
    def _load_pep_list(self) -> set:
        """
        Load PEP (Politically Exposed Persons) list
        
        In production, integrate with:
        - World-Check (Refinitiv)
        - Dow Jones Watchlist
        - LexisNexis
        - PEP databases
        
        For now, simulated list.
        """
        logger.info("📋 Using simulated PEP list")
        return {
            "CUST-PEP-001", "CUST-PEP-002", "CUST-PEP-003",
            "john doe politician", "jane smith minister"
        }
    
    def refresh_sanctions_list(self) -> bool:
        """
        Force refresh of ALL sanctions lists (OFAC + UN + EU)
        Returns True if successful, False otherwise
        """
        logger.info("🔄 Force refreshing all sanctions lists (OFAC + UN + EU)...")
        self._sanctions_cache = None
        self.sanctions_last_update = None
        
        new_list = self._load_all_sanctions_lists()
        
        if new_list and len(new_list) > 100:  # Sanity check
            self.sanctions_list = new_list
            logger.success(f"✅ All sanctions lists refreshed: {len(new_list)} total entries")
            return True
        else:
            logger.error("❌ Failed to refresh sanctions lists")
            return False
    
    def get_sanctions_stats(self) -> dict:
        """
        Get statistics about loaded sanctions lists
        """
        return {
            "total_entities": len(self.sanctions_list),
            "last_update": self.sanctions_last_update.isoformat() if self.sanctions_last_update else None,
            "cache_age_hours": (datetime.utcnow() - self.sanctions_last_update).total_seconds() / 3600 if self.sanctions_last_update else None,
            "sources": ["OFAC (US)", "UN Security Council", "European Union"],
            "cache_ttl_hours": self.sanctions_cache_ttl.total_seconds() / 3600
        }
    
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Enrich transaction with additional data"""
        start_time = time.time()
        
        findings = []
        patterns_detected = []
        risk_score = 0.0
        suspicious = False
        
        # Check high-risk corntries
        if transaction.country_origin in self.high_risk_countries:
            findings.append(f"Transaction originates from high-risk country: {transaction.country_origin}")
            patterns_detected.append("high_risk_country_origin")
            risk_score = max(risk_score, 0.7)
            suspicious = True
        
        if transaction.country_destination in self.high_risk_countries:
            findings.append(f"Transaction destined for high-risk country: {transaction.country_destination}")
            patterns_detected.append("high_risk_country_destination")
            risk_score = max(risk_score, 0.7)
            suspicious = True
        
        # Check sanctions lists
        # Check both by ID and by namand (if available)
        sender_sanctioned = self._check_sanctions(
            transaction.sender_id,
            transaction.sender_name
        )
        receiver_sanctioned = self._check_sanctions(
            transaction.receiver_id,
            transaction.receiver_name
        )
        
        if sender_sanctioned:
            findings.append(f"Sender is on sanctions list: {transaction.sender_id}")
            patterns_detected.append("sanctioned_sender")
            risk_score = 1.0
            suspicious = True
        
        if receiver_sanctioned:
            findings.append(f"Receiver is on sanctions list: {transaction.receiver_id}")
            patterns_detected.append("sanctioned_receiver")
            risk_score = 1.0
            suspicious = True
        
        # Check PEP status
        sender_pep = self._check_pep(
            transaction.sender_id,
            transaction.sender_name
        )
        receiver_pep = self._check_pep(
            transaction.receiver_id,
            transaction.receiver_name
        )
        
        if sender_pep:
            findings.append(f"Sender is a Politically Exposed Person (PEP): {transaction.sender_id}")
            patterns_detected.append("pep_sender")
            risk_score = max(risk_score, 0.6)
        
        if receiver_pep:
            findings.append(f"Receiver is a Politically Exposed Person (PEP): {transaction.receiver_id}")
            patterns_detected.append("pep_receiver")
            risk_score = max(risk_score, 0.6)
        
        # Enrich transaction data
        transaction.enriched_data["sender_sanctioned"] = sender_sanctioned
        transaction.enriched_data["receiver_sanctioned"] = receiver_sanctioned
        transaction.enriched_data["sender_pep"] = sender_pep
        transaction.enriched_data["receiver_pep"] = receiver_pep
        transaction.enriched_data["high_risk_origin"] = transaction.country_origin in self.high_risk_countries
        transaction.enriched_data["high_risk_destination"] = transaction.country_destination in self.high_risk_countries
        
        execution_time = time.time() - start_time
        
        explanation = f"Enrichment complete. Added contextual information. " + (
            f"Detected {len(patterns_detected)} risk indicators." if patterns_detected else "No special risk indicators."
        )
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=execution_time,
            suspicious=suspicious,
            confidence=0.98,
            risk_score=risk_score,
            findings=findings,
            patterns_detected=patterns_detected,
            explanation=explanation,
            evidence={
                "sanctions_checked": True,
                "pep_checked": True,
                "country_risk_assessed": True
            },
            recommended_action="escalate" if suspicious and risk_score >= 0.9 else "continue",
            alert_should_be_created=suspicious and risk_score >= 0.7
        )
    
    def _check_sanctions(self, customer_id: str, customer_name: Optional[str]) -> bool:
        """
        Check if customer is in sanctions list
        Checks both by ID and by name (normalized)
        """
        # Check by ID
        if customer_id and customer_id.lower() in self.sanctions_list:
            return True
        
        # Check by namand (if available)
        if customer_name:
            normalized_name = customer_name.lower().strip()
            if normalized_name in self.sanctions_list:
                return True
            
            # Check partial matches (for withpornd names)
            for sanctioned in self.sanctions_list:
                if len(sanctioned) > 5:  # Avoid short matches
                    if sanctioned in normalized_name or normalized_name in sanctioned:
                        logger.warning(f"⚠️  Potential sanction match: '{customer_name}' ~= '{sanctioned}'")
                        return True
        
        return False
    
    def _check_pep(self, customer_id: str, customer_name: Optional[str]) -> bool:
        """
        Check if customer is a PEP (Politically Exposed Person)
        """
        # Check by ID
        if customer_id and customer_id in self.pep_list:
            return True
        
        # Check by name
        if customer_name:
            normalized_name = customer_name.lower().strip()
            if normalized_name in self.pep_list:
                return True
        
        return False


class CustomerProfileAgent(BaseAgent):
    """
    Agent that enriches transactions with customer profile information
    and historical behavior patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="customer_profile_agent",
            agent_type="enrichment",
            config=config
        )
        
        # In production, this world thatry a customer databaif
        self.customer_cache: Dict[str, Customer] = {}
    
    def _get_customer_profile(self, customer_id: str) -> Optional[Customer]:
        """Get or creatand customer profiland (simulated)"""
        if customer_id in self.customer_cache:
            return self.customer_cache[customer_id]
        
        # Simulatand customer profiland creation
        # In production, this world thatry databaif
        return None
    
    async def analyze(self, transaction: Transaction, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        """Enrich with customer profiland data"""
        start_time = time.time()
        
        findings = []
        patterns_detected = []
        risk_score = 0.0
        suspicious = False
        
        # Get customer profiles from context (if proviofd)
        sender_profile = None
        receiver_profile = None
        
        if context:
            sender_profile = context.get("sender_profile")
            receiver_profile = context.get("receiver_profile")
        
        # Check ifnofr profile
        if sender_profile:
            transaction.enriched_data["sender_risk_rating"] = sender_profile.risk_rating.value
            transaction.enriched_data["sender_kyc_status"] = sender_profile.kyc_status
            transaction.enriched_data["sender_is_pep"] = sender_profile.is_pep
            
            if sender_profile.risk_rating in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                findings.append(f"Sender has high risk rating: {sender_profile.risk_rating.value}")
                risk_score = max(risk_score, 0.7)
                suspicious = True
            
            if sender_profile.kyc_status != "verified":
                findings.append(f"Sender KYC status: {sender_profile.kyc_status}")
                risk_score = max(risk_score, 0.5)
        
        # Check receiver profile
        if receiver_profile:
            transaction.enriched_data["receiver_risk_rating"] = receiver_profile.risk_rating.value
            transaction.enriched_data["receiver_kyc_status"] = receiver_profile.kyc_status
            transaction.enriched_data["receiver_is_pep"] = receiver_profile.is_pep
            
            if receiver_profile.risk_rating in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                findings.append(f"Receiver has high risk rating: {receiver_profile.risk_rating.value}")
                risk_score = max(risk_score, 0.7)
                suspicious = True
        
        execution_time = time.time() - start_time
        
        explanation = "Customer profile enrichment complete."
        
        return AgentResult(
            agent_name=self.agent_id,
            agent_type=self.agent_type,
            execution_time=execution_time,
            suspicious=suspicious,
            confidence=0.90,
            risk_score=risk_score,
            findings=findings,
            patterns_detected=patterns_detected,
            explanation=explanation,
            evidence={
                "sender_profile_retrieved": sender_profile is not None,
                "receiver_profile_retrieved": receiver_profile is not None
            },
            recommended_action="continue",
            alert_should_be_created=False
        )

