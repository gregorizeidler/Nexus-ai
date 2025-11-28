"""
🏦 SWIFT MESSAGE PARSER
Parse SWIFT MT messages for AML analysis
"""
import re
from typing import Dict, Any, Optional
from decimal import Decimal
from loguru import logger


class SWIFTParser:
    """
    Parser for SWIFT MT messages (wire transfers)
    """
    
    def __init__(self):
        logger.success("🏦 SWIFT Parser initialized")
    
    def parse_mt103(self, message: str) -> Dict[str, Any]:
        """
        Parse SWIFT MT103 (Single Customer Credit Transfer)
        """
        fields = self._extract_fields(message)
        
        # Field 20: Transaction Reference
        transaction_ref = fields.get('20', '')
        
        # Field 32A: Valuand Date, Currency, Amornt
        amount_field = fields.get('32A', '')
        value_date, currency, amount = self._parse_32A(amount_field)
        
        # Field 50K: Orofring Customer
        sender = self._parse_party_field(fields.get('50K', ''))
        
        # Field 59: Beneficiary Customer
        receiver = self._parse_party_field(fields.get('59', ''))
        
        # Field 71A: oftails of Charges
        charges = fields.get('71A', '')
        
        return {
            'message_type': 'MT103',
            'transaction_id': transaction_ref,
            'value_date': value_date,
            'currency': currency,
            'amount': amount,
            'sender': sender,
            'receiver': receiver,
            'charges': charges,
            'raw_message': message
        }
    
    def parse_mt202(self, message: str) -> Dict[str, Any]:
        """
        Parse SWIFT MT202 (General Financial Institution Transfer)
        """
        fields = self._extract_fields(message)
        
        return {
            'message_type': 'MT202',
            'transaction_id': fields.get('20', ''),
            'amount': self._parse_32A(fields.get('32A', ''))[2],
            'ordering_institution': fields.get('52A', ''),
            'beneficiary_institution': fields.get('58A', '')
        }
    
    def _extract_fields(self, message: str) -> Dict[str, str]:
        """Extract SWIFT fields from message"""
        fields = {}
        
        # Pattern: :FIELD_NUMBER:CONTENT
        pattern = r':(\d{2}[A-Z]?):(.*?)(?=:\d{2}[A-Z]?:|$)'
        
        matches = re.findall(pattern, message, re.DOTALL)
        
        for field_number, content in matches:
            fields[field_number] = content.strip()
        
        return fields
    
    def _parse_32A(self, field_32A: str) -> tuple:
        """
        Parse field 32A: Date, Currency, Amount
        Format: YYMMDDCCCAMOUNT (e.g., 240101USD50000,00)
        """
        if not field_32A or len(field_32A) < 9:
            return ('', '', 0.0)
        
        date = field_32A[:6]  # YYMMDD
        currency = field_32A[6:9]  # CCC
        amount_str = field_32A[9:].replace(',', '.')
        
        try:
            amount = float(amount_str)
        except:
            amount = 0.0
        
        return (date, currency, amount)
    
    def _parse_party_field(self, field: str) -> Dict[str, str]:
        """
        Parse party fields (50K, 59, etc)
        Format: /ACCOUNT\nNAME\nADDRESS
        """
        lines = field.split('\n')
        
        account = ''
        name = ''
        address = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('/'):
                account = line[1:]
            elif i == 0 or (i == 1 and account):
                name = line
            else:
                address.append(line)
        
        return {
            'account': account,
            'name': name,
            'address': ' '.join(address)
        }


class ISO8583Parser:
    """
    Real ISO 8583 message parser for card transactions
    
    Supports:
    - MTI extraction (Message Type Indicator)
    - Primary and secondary bitmap parsing
    - Variable and fixed-length field parsing
    - Common card transaction fields (2-128)
    - Multiple encoding formats (ASCII, BCD, Binary)
    """
    
    # Field specifications: (length_type, max_length, data_type)
    # length_type: 'FIXED', 'LLVAR' (2-digit length), 'LLLVAR' (3-digit length)
    # data_type: 'n' (numeric), 'an' (alphanumeric), 'ans' (alphanumeric+special), 'b' (binary)
    FIELD_SPECS = {
        2: ('LLVAR', 19, 'n'),      # Primary Account Number (PAN)
        3: ('FIXED', 6, 'n'),        # Processing Code
        4: ('FIXED', 12, 'n'),       # Transaction Amount
        5: ('FIXED', 12, 'n'),       # Settlement Amount
        7: ('FIXED', 10, 'n'),       # Transmission Date/Time (MMDDhhmmss)
        11: ('FIXED', 6, 'n'),       # System Trace Audit Number (STAN)
        12: ('FIXED', 6, 'n'),       # Local Transaction Time (hhmmss)
        13: ('FIXED', 4, 'n'),       # Local Transaction Date (MMDD)
        14: ('FIXED', 4, 'n'),       # Card Expiration Date (YYMM)
        18: ('FIXED', 4, 'n'),       # Merchant Category Code (MCC)
        19: ('FIXED', 3, 'n'),       # Acquiring Institution Country Code
        22: ('FIXED', 3, 'n'),       # POS Entry Mode
        25: ('FIXED', 2, 'n'),       # POS Condition Code
        32: ('LLVAR', 11, 'n'),      # Acquiring Institution ID
        33: ('LLVAR', 11, 'n'),      # Forwarding Institution ID
        35: ('LLVAR', 37, 'ans'),    # Track 2 Data
        37: ('FIXED', 12, 'an'),     # Retrieval Reference Number
        38: ('FIXED', 6, 'an'),      # Authorization ID Response
        39: ('FIXED', 2, 'an'),      # Response Code
        41: ('FIXED', 8, 'ans'),     # Card Acceptor Terminal ID
        42: ('FIXED', 15, 'ans'),    # Card Acceptor ID (Merchant ID)
        43: ('FIXED', 40, 'ans'),    # Card Acceptor Name/Location
        49: ('FIXED', 3, 'an'),      # Transaction Currency Code
        50: ('FIXED', 3, 'an'),      # Settlement Currency Code
        52: ('FIXED', 8, 'b'),       # PIN Data
        54: ('LLLVAR', 120, 'an'),   # Additional Amounts
        55: ('LLLVAR', 255, 'b'),    # ICC Data (EMV)
        60: ('LLLVAR', 999, 'ans'),  # Reserved Private
        61: ('LLLVAR', 999, 'ans'),  # Reserved Private
        62: ('LLLVAR', 999, 'ans'),  # Reserved Private
        63: ('LLLVAR', 999, 'ans'),  # Reserved Private
        90: ('FIXED', 42, 'n'),      # Original Data Elements
        95: ('FIXED', 42, 'an'),     # Replacement Amounts
        100: ('LLVAR', 11, 'n'),     # Receiving Institution ID
        102: ('LLVAR', 28, 'ans'),   # Account ID 1
        103: ('LLVAR', 28, 'ans'),   # Account ID 2
        123: ('LLLVAR', 999, 'ans'), # POS Data Extended
        125: ('LLVAR', 50, 'ans'),   # Network Management Info
        128: ('FIXED', 8, 'b'),      # Message Authentication Code (MAC)
    }
    
    def __init__(self):
        logger.success("💳 ISO 8583 Parser initialized (Real Implementation)")
    
    def parse(self, message: bytes) -> Dict[str, Any]:
        """
        Parse ISO 8583 message (ASCII encoding)
        
        Args:
            message: Raw ISO 8583 message bytes
            
        Returns:
            Dictionary with MTI and all present fields
        """
        try:
            # Convert to string for ASCII parsing
            msg_str = message.decode('ascii') if isinstance(message, bytes) else message
            
            position = 0
            
            # 1. Extract MTI (4 bytes)
            mti = msg_str[position:position+4]
            position += 4
            
            logger.debug(f"MTI: {mti}")
            
            # 2. Extract Primary Bitmap (16 hex chars = 64 bits)
            primary_bitmap_hex = msg_str[position:position+16]
            primary_bitmap = int(primary_bitmap_hex, 16)
            position += 16
            
            logger.debug(f"Primary Bitmap: {primary_bitmap_hex}")
            
            # 3. Check for Secondary Bitmap (bit 1 set)
            secondary_bitmap = 0
            if primary_bitmap & (1 << 63):
                secondary_bitmap_hex = msg_str[position:position+16]
                secondary_bitmap = int(secondary_bitmap_hex, 16)
                position += 16
                logger.debug(f"Secondary Bitmap: {secondary_bitmap_hex}")
            
            # 4. Parse fields
            fields = {}
            
            for field_num in range(2, 129):  # Field 1 is bitmap itself
                if self._is_field_present(field_num, primary_bitmap, secondary_bitmap):
                    field_value, bytes_consumed = self._parse_field(
                        field_num, 
                        msg_str[position:]
                    )
                    fields[field_num] = field_value
                    position += bytes_consumed
                    
                    logger.debug(f"Field {field_num}: {field_value}")
            
            # 5. Build result
            result = {
                'mti': mti,
                'mti_description': self._get_mti_description(mti),
                'fields': fields,
                'parsed_fields': self._format_fields_for_aml(fields),
                'raw_message': msg_str
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse ISO 8583 message: {e}")
            return {
                'error': str(e),
                'mti': None,
                'fields': {}
            }
    
    def _is_field_present(self, field_num: int, primary_bitmap: int, secondary_bitmap: int) -> bool:
        """Check if field is present in bitmap"""
        if field_num <= 64:
            # Check primary bitmap (bits 1-64, but field 1 is the bitmap itself)
            return bool(primary_bitmap & (1 << (64 - field_num)))
        else:
            # Check secondary bitmap (fields 65-128)
            return bool(secondary_bitmap & (1 << (128 - field_num)))
    
    def _parse_field(self, field_num: int, data: str) -> tuple:
        """
        Parse a single field
        
        Returns:
            (field_value, bytes_consumed)
        """
        if field_num not in self.FIELD_SPECS:
            # Unknown field - skip
            return (None, 0)
        
        length_type, max_length, data_type = self.FIELD_SPECS[field_num]
        position = 0
        
        # Determine field length
        if length_type == 'FIXED':
            field_length = max_length
        elif length_type == 'LLVAR':
            # 2-digit length prefix
            length_str = data[position:position+2]
            field_length = int(length_str)
            position += 2
        elif length_type == 'LLLVAR':
            # 3-digit length prefix
            length_str = data[position:position+3]
            field_length = int(length_str)
            position += 3
        else:
            return (None, 0)
        
        # Extract field data
        field_value = data[position:position+field_length]
        total_consumed = position + field_length
        
        return (field_value, total_consumed)
    
    def _get_mti_description(self, mti: str) -> str:
        """Get human-readable MTI description"""
        mti_map = {
            '0100': 'Authorization Request',
            '0110': 'Authorization Response',
            '0120': 'Authorization Advice',
            '0121': 'Authorization Advice Repeat',
            '0200': 'Financial Transaction Request',
            '0210': 'Financial Transaction Response',
            '0220': 'Financial Transaction Advice',
            '0221': 'Financial Transaction Advice Repeat',
            '0400': 'Reversal Request',
            '0410': 'Reversal Response',
            '0420': 'Reversal Advice',
            '0421': 'Reversal Advice Repeat',
            '0800': 'Network Management Request',
            '0810': 'Network Management Response',
        }
        return mti_map.get(mti, f'Unknown MTI: {mti}')
    
    def _format_fields_for_aml(self, fields: Dict[int, str]) -> Dict[str, Any]:
        """
        Format parsed fields for AML analysis
        """
        formatted = {}
        
        # Field 2: PAN (Primary Account Number)
        if 2 in fields:
            pan = fields[2]
            formatted['card_number'] = self._mask_pan(pan)
            formatted['card_bin'] = pan[:6] if len(pan) >= 6 else pan
            formatted['card_last4'] = pan[-4:] if len(pan) >= 4 else pan
        
        # Field 4: Transaction Amount (in cents)
        if 4 in fields:
            amount_cents = int(fields[4])
            formatted['amount'] = amount_cents / 100.0
        
        # Field 7: Transmission Date/Time
        if 7 in fields:
            dt = fields[7]  # MMDDhhmmss
            formatted['transmission_datetime'] = f"{dt[0:2]}/{dt[2:4]} {dt[4:6]}:{dt[6:8]}:{dt[8:10]}"
        
        # Field 11: STAN (System Trace Audit Number)
        if 11 in fields:
            formatted['trace_number'] = fields[11]
        
        # Field 12: Local Transaction Time
        if 12 in fields:
            time = fields[12]  # hhmmss
            formatted['local_time'] = f"{time[0:2]}:{time[2:4]}:{time[4:6]}"
        
        # Field 13: Local Transaction Date
        if 13 in fields:
            date = fields[13]  # MMDD
            formatted['local_date'] = f"{date[0:2]}/{date[2:4]}"
        
        # Field 18: Merchant Category Code (MCC)
        if 18 in fields:
            mcc = fields[18]
            formatted['merchant_category'] = mcc
            formatted['merchant_category_name'] = self._get_mcc_description(mcc)
        
        # Field 19: Acquiring Country
        if 19 in fields:
            formatted['acquiring_country'] = fields[19]
        
        # Field 22: POS Entry Mode
        if 22 in fields:
            formatted['pos_entry_mode'] = fields[22]
            formatted['pos_entry_description'] = self._get_pos_entry_mode(fields[22])
        
        # Field 25: POS Condition Code
        if 25 in fields:
            formatted['pos_condition'] = fields[25]
        
        # Field 32: Acquiring Institution ID
        if 32 in fields:
            formatted['acquiring_institution'] = fields[32]
        
        # Field 37: Retrieval Reference Number
        if 37 in fields:
            formatted['reference_number'] = fields[37]
        
        # Field 38: Authorization ID
        if 38 in fields:
            formatted['authorization_id'] = fields[38]
        
        # Field 39: Response Code
        if 39 in fields:
            response_code = fields[39]
            formatted['response_code'] = response_code
            formatted['response_description'] = self._get_response_code_description(response_code)
        
        # Field 41: Terminal ID
        if 41 in fields:
            formatted['terminal_id'] = fields[41].strip()
        
        # Field 42: Merchant ID
        if 42 in fields:
            formatted['merchant_id'] = fields[42].strip()
        
        # Field 43: Merchant Name/Location
        if 43 in fields:
            formatted['merchant_location'] = fields[43].strip()
        
        # Field 49: Transaction Currency
        if 49 in fields:
            formatted['currency_code'] = fields[49]
        
        # Field 55: ICC/EMV Data (chip card)
        if 55 in fields:
            formatted['emv_data_present'] = True
            formatted['is_chip_transaction'] = True
        else:
            formatted['is_chip_transaction'] = False
        
        return formatted
    
    def _mask_pan(self, pan: str) -> str:
        """Mask PAN for security (show first 6 and last 4)"""
        if len(pan) <= 10:
            return pan
        return f"{pan[:6]}{'*' * (len(pan) - 10)}{pan[-4:]}"
    
    def _get_mcc_description(self, mcc: str) -> str:
        """Get Merchant Category Code description (high-risk categories)"""
        mcc_map = {
            '6211': 'Securities Brokers/Dealers',
            '6051': 'Cryptocurrency Exchanges',
            '7995': 'Gambling - Casinos',
            '7801': 'Gambling - Online',
            '5933': 'Pawn Shops',
            '5094': 'Precious Metals/Jewelry',
            '4829': 'Money Transfer',
            '6012': 'Financial Institutions',
            '7299': 'Adult Entertainment',
            '5912': 'Drug Stores/Pharmacies',
        }
        return mcc_map.get(mcc, f'MCC {mcc}')
    
    def _get_pos_entry_mode(self, mode: str) -> str:
        """Get POS entry mode description"""
        mode_map = {
            '000': 'Unknown',
            '010': 'Manual Key Entry',
            '020': 'Magnetic Stripe',
            '051': 'Chip Card (EMV)',
            '071': 'Contactless Chip',
            '072': 'Contactless Magnetic Stripe',
            '090': 'E-Commerce',
        }
        return mode_map.get(mode, f'Mode {mode}')
    
    def _get_response_code_description(self, code: str) -> str:
        """Get response code description"""
        code_map = {
            '00': 'Approved',
            '01': 'Refer to Card Issuer',
            '05': 'Do Not Honor',
            '14': 'Invalid Card Number',
            '41': 'Lost Card',
            '43': 'Stolen Card',
            '51': 'Insufficient Funds',
            '54': 'Expired Card',
            '57': 'Transaction Not Permitted',
            '61': 'Exceeds Withdrawal Limit',
            '62': 'Restricted Card',
            '63': 'Security Violation',
        }
        return code_map.get(code, f'Response {code}')
    
    def extract_aml_risk_indicators(self, parsed_msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract AML risk indicators from parsed ISO 8583 message
        """
        fields = parsed_msg.get('parsed_fields', {})
        risk_flags = []
        risk_score = 0.0
        
        # Amount analysis
        amount = fields.get('amount', 0.0)
        if 9000 <= amount <= 9999:
            risk_flags.append('STRUCTURING_THRESHOLD')
            risk_score += 0.4
        
        # Time analysis (off-hours transactions)
        local_time = fields.get('local_time', '')
        if local_time and ':' in local_time:
            try:
                hour = int(local_time.split(':')[0])
                if hour < 6 or hour > 22:
                    risk_flags.append('OFF_HOURS_TRANSACTION')
                    risk_score += 0.2
            except (ValueError, IndexError):
                pass
        
        # High-risk MCC
        mcc = fields.get('merchant_category', '')
        high_risk_mccs = ['7995', '7801', '6051', '4829', '5933']
        if mcc in high_risk_mccs:
            risk_flags.append(f'HIGH_RISK_MCC_{mcc}')
            risk_score += 0.3
        
        # Manual entry (higher fraud risk)
        if fields.get('pos_entry_description') == 'Manual Key Entry':
            risk_flags.append('MANUAL_CARD_ENTRY')
            risk_score += 0.15
        
        # Card present vs not present
        if not fields.get('is_chip_transaction', False):
            risk_flags.append('NON_CHIP_TRANSACTION')
            risk_score += 0.1
        
        # Failed transactions (testing cards)
        response = fields.get('response_code', '00')
        if response in ['14', '41', '43', '54']:
            risk_flags.append(f'SUSPICIOUS_RESPONSE_{response}')
            risk_score += 0.25
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_flags': risk_flags,
            'risk_level': 'HIGH' if risk_score >= 0.7 else 'MEDIUM' if risk_score >= 0.4 else 'LOW'
        }

