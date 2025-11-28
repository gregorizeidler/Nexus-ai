"""
Generate synthetic transaction data for testing the AML/CFT system.
"""
import random
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict
from faker import Faker

fake = Faker()


class SyntheticDataGenerator:
    """Generate realistic synthetic financial transactions"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        Faker.seed(seed)
        
        self.currencies = ["USD", "EUR", "GBP", "BRL", "JPY"]
        self.transaction_types = [
            "wire_transfer", "cash_deposit", "cash_withdrawal", 
            "ach", "card_payment", "international"
        ]
        
        # Generate customer pool
        self.customers = self._generate_customers(100)
        
        # High-risk entities for testing
        self.high_risk_customers = random.sample(self.customers, 10)
        self.pep_customers = random.sample(self.customers, 5)
        self.sanctioned_customers = random.sample(self.customers, 2)
    
    def _generate_customers(self, count: int) -> List[str]:
        """Generate customer IDs"""
        return [f"CUST-{i:06d}" for i in range(1, count + 1)]
    
    def generate_normal_transaction(self) -> Dict:
        """Generate a normal, non-suspicious transaction"""
        sender = random.choice(self.customers)
        receiver = random.choice([c for c in self.customers if c != sender])
        
        amount = random.choice([
            round(random.uniform(100, 1000), 2),
            round(random.uniform(1000, 5000), 2),
            round(random.uniform(50, 500), 2),
        ])
        
        return {
            "transaction_id": f"TXN-{fake.uuid4()[:8].upper()}",
            "timestamp": (datetime.utcnow() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )).isoformat() + "Z",
            "amount": amount,
            "currency": random.choice(self.currencies),
            "transaction_type": random.choice(self.transaction_types),
            "sender_id": sender,
            "sender_account": f"ACC-{fake.uuid4()[:8].upper()}",
            "receiver_id": receiver,
            "receiver_account": f"ACC-{fake.uuid4()[:8].upper()}",
            "country_origin": fake.country_code(),
            "country_destination": fake.country_code(),
            "description": fake.sentence(),
            "channel": random.choice(["online", "mobile", "branch", "atm"])
        }
    
    def generate_high_value_transaction(self) -> Dict:
        """Generate a high-value transaction (potential trigger)"""
        txn = self.generate_normal_transaction()
        txn["amount"] = round(random.uniform(10000, 50000), 2)
        txn["transaction_type"] = "wire_transfer"
        return txn
    
    def generate_structuring_pattern(self, count: int = 5) -> List[Dict]:
        """Generate structuring pattern (multiple transactions just below threshold)"""
        sender = random.choice(self.customers)
        receiver = random.choice([c for c in self.customers if c != sender])
        
        transactions = []
        base_time = datetime.utcnow() - timedelta(hours=random.randint(1, 24))
        
        for i in range(count):
            # Amounts just below 10,000 threshold
            amount = round(random.uniform(9500, 9900), 2)
            
            transactions.append({
                "transaction_id": f"TXN-{fake.uuid4()[:8].upper()}",
                "timestamp": (base_time + timedelta(minutes=i*15)).isoformat() + "Z",
                "amount": amount,
                "currency": "USD",
                "transaction_type": "wire_transfer",
                "sender_id": sender,
                "sender_account": f"ACC-{fake.uuid4()[:8].upper()}",
                "receiver_id": receiver,
                "receiver_account": f"ACC-{fake.uuid4()[:8].upper()}",
                "country_origin": "US",
                "country_destination": "US",
                "description": f"Payment {i+1}",
                "channel": "online"
            })
        
        return transactions
    
    def generate_smurfing_pattern(self, count: int = 10) -> List[Dict]:
        """Generate smurfing pattern (many small transactions)"""
        sender = random.choice(self.customers)
        receivers = random.sample(self.customers, min(count, len(self.customers)))
        
        transactions = []
        base_time = datetime.utcnow() - timedelta(hours=random.randint(1, 12))
        
        for i, receiver in enumerate(receivers):
            amount = round(random.uniform(500, 3000), 2)
            
            transactions.append({
                "transaction_id": f"TXN-{fake.uuid4()[:8].upper()}",
                "timestamp": (base_time + timedelta(minutes=i*5)).isoformat() + "Z",
                "amount": amount,
                "currency": "USD",
                "transaction_type": random.choice(["ach", "wire_transfer"]),
                "sender_id": sender,
                "sender_account": f"ACC-{fake.uuid4()[:8].upper()}",
                "receiver_id": receiver,
                "receiver_account": f"ACC-{fake.uuid4()[:8].upper()}",
                "country_origin": "US",
                "country_destination": "US",
                "description": "Transfer",
                "channel": "online"
            })
        
        return transactions
    
    def generate_high_risk_country_transaction(self) -> Dict:
        """Generate transaction involving high-risk country"""
        txn = self.generate_normal_transaction()
        high_risk_countries = ["IR", "KP", "SY", "AF", "IQ"]
        txn["country_origin"] = random.choice(high_risk_countries)
        txn["amount"] = round(random.uniform(5000, 20000), 2)
        return txn
    
    def generate_pep_transaction(self) -> Dict:
        """Generate transaction involving PEP"""
        txn = self.generate_normal_transaction()
        txn["sender_id"] = "CUST-PEP-001"  # Known PEP
        txn["amount"] = round(random.uniform(5000, 15000), 2)
        return txn
    
    def generate_sanctioned_transaction(self) -> Dict:
        """Generate transaction involving sanctioned entity"""
        txn = self.generate_normal_transaction()
        txn["sender_id"] = "CUST-SANCT-001"  # Known sanctioned entity
        txn["amount"] = round(random.uniform(1000, 10000), 2)
        return txn
    
    def generate_dataset(
        self, 
        normal_count: int = 1000,
        suspicious_percentage: float = 0.05
    ) -> List[Dict]:
        """
        Generate a complete dataset with normal and suspicious transactions.
        
        Args:
            normal_count: Number of normal transactions
            suspicious_percentage: Percentage of suspicious transactions
            
        Returns:
            List of transaction dictionaries
        """
        transactions = []
        
        # Generate normal transactions
        print(f"Generating {normal_count} normal transactions...")
        for _ in range(normal_count):
            transactions.append(self.generate_normal_transaction())
        
        # Calculate suspicious transaction count
        suspicious_count = int(normal_count * suspicious_percentage)
        
        print(f"Generating {suspicious_count} suspicious transactions...")
        
        # Generate different types of suspicious patterns
        patterns = {
            "high_value": suspicious_count // 5,
            "structuring": suspicious_count // 10,
            "smurfing": suspicious_count // 10,
            "high_risk_country": suspicious_count // 5,
            "pep": suspicious_count // 10,
            "sanctioned": suspicious_count // 10,
        }
        
        # High value
        for _ in range(patterns["high_value"]):
            transactions.append(self.generate_high_value_transaction())
        
        # Structuring patterns
        for _ in range(patterns["structuring"]):
            transactions.extend(self.generate_structuring_pattern(random.randint(3, 7)))
        
        # Smurfing patterns
        for _ in range(patterns["smurfing"]):
            transactions.extend(self.generate_smurfing_pattern(random.randint(5, 15)))
        
        # High risk country
        for _ in range(patterns["high_risk_country"]):
            transactions.append(self.generate_high_risk_country_transaction())
        
        # PEP transactions
        for _ in range(patterns["pep"]):
            transactions.append(self.generate_pep_transaction())
        
        # Sanctioned entity transactions
        for _ in range(patterns["sanctioned"]):
            transactions.append(self.generate_sanctioned_transaction())
        
        # Shuffle transactions
        random.shuffle(transactions)
        
        print(f"Total transactions generated: {len(transactions)}")
        
        return transactions


def main():
    """Main function to generate and save synthetic data"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic AML transaction data")
    parser.add_argument("--num-transactions", type=int, default=1000, help="Number of normal transactions")
    parser.add_argument("--suspicious-pct", type=float, default=0.05, help="Percentage of suspicious transactions")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_transactions.json", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AML/CFT Synthetic Data Generator")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=args.seed)
    transactions = generator.generate_dataset(
        normal_count=args.num_transactions,
        suspicious_percentage=args.suspicious_pct
    )
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(transactions, f, indent=2)
    
    print(f"\n✅ Data saved to: {args.output}")
    print(f"   Total transactions: {len(transactions)}")
    print(f"   Expected suspicious: ~{int(args.num_transactions * args.suspicious_pct * 1.5)}")
    
    # Generate summary statistics
    print("\n📊 Dataset Summary:")
    print(f"   Normal transactions: {args.num_transactions}")
    print(f"   Suspicious patterns: ~{int(args.num_transactions * args.suspicious_pct)}")
    print(f"   Total size: {len(transactions)}")
    
    # Sample transactions
    print("\n🔍 Sample Transactions:")
    for txn in random.sample(transactions, min(3, len(transactions))):
        print(f"   {txn['transaction_id']}: {txn['amount']} {txn['currency']} - {txn['transaction_type']}")
    
    print("\n" + "=" * 60)
    print("✨ Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

