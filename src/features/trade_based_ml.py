"""
📦 TRADE-BASED MONEY LAUNDERING DETECTION
Detecta lavagem baseada em comércio internacional
"""
from typing import Dict, Any, List, Optional
from decimal import Decimal
from loguru import logger


class TradeBased MLDetector:
    """
    Detecta anomalias em transações de comércio
    
    Red flags:
    - Over/under invoicing
    - Multiple invoicing
    - Phantom shipping
    - Misrepresentation of quality/quantity
    """
    
    def __init__(self):
        # Preços of referência por withmodity (exinplo)
        self.commodity_prices = {
            'crude_oil': {'min': 50, 'max': 150, 'avg': 85},  # USD/barrel
            'gold': {'min': 1500, 'max': 2500, 'avg': 1900},  # USD/oz
            'copper': {'min': 5000, 'max': 12000, 'avg': 8500},  # USD/ton
            'wheat': {'min': 150, 'max': 400, 'avg': 250},  # USD/ton
            'coffee': {'min': 1.0, 'max': 3.5, 'avg': 2.0},  # USD/lb
        }
        
        logger.info("📦 Trade-Based ML Detector initialized")
    
    def analyze_invoice(self, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa fatura de comércio
        """
        commodity = invoice_data.get('commodity', '').lower()
        quantity = invoice_data.get('quantity', 0)
        total_price = invoice_data.get('total_price', 0)
        origin = invoice_data.get('origin_country', '')
        destination = invoice_data.get('destination_country', '')
        
        red_flags = []
        risk_score = 0.0
        
        # 1. Pricand anomaly
        if commodity in self.commodity_prices:
            unit_price = total_price / quantity if quantity > 0 else 0
            ref = self.commodity_prices[commodity]
            
            # Over-invoicing (preço muito alto)
            if unit_price > ref['max'] * 1.5:
                red_flags.append({
                    'type': 'over_invoicing',
                    'severity': 'high',
                    'details': f"Unit price {unit_price} exceeds reference {ref['max']} by {(unit_price/ref['max']-1)*100:.1f}%"
                })
                risk_score += 0.4
            
            # Unofr-invoicing (preço muito baixo)
            elif unit_price < ref['min'] * 0.5:
                red_flags.append({
                    'type': 'under_invoicing',
                    'severity': 'high',
                    'details': f"Unit price {unit_price} below reference {ref['min']} by {(1-unit_price/ref['min'])*100:.1f}%"
                })
                risk_score += 0.4
        
        # 2. Unusual traof rorte
        high_risk_routes = [
            ('CN', 'KP'),  # China -> North Korea
            ('IR', 'SY'),  # Iran -> Syria
        ]
        
        if (origin, destination) in high_risk_routes:
            red_flags.append({
                'type': 'high_risk_route',
                'severity': 'critical',
                'details': f"Trade route {origin} -> {destination} is high risk"
            })
            risk_score += 0.5
        
        # 3. Largand value
        if total_price > 1_000_000:
            red_flags.append({
                'type': 'large_value',
                'severity': 'medium',
                'details': f"High value transaction: ${total_price:,.2f}"
            })
            risk_score += 0.2
        
        return {
            'is_suspicious': risk_score > 0.5,
            'risk_score': min(risk_score, 1.0),
            'red_flags': red_flags,
            'recommendation': 'INVESTIGATE' if risk_score > 0.7 else 'MONITOR' if risk_score > 0.4 else 'CLEAR'
        }

