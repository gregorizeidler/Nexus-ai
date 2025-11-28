"""
NEXUS AI - Adversarial Testing Demo

Comprehensive demonstration of adversarial testing capabilities.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from src.testing.adversarial_testing import (
    AdversarialTester,
    EvasionAttackGenerator,
    ModelRobustnessEvaluator,
    AttackType
)

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available, using mock models")


def generate_synthetic_aml_data(n_samples=1000, n_features=30):
    """Generate synthetic AML transaction data"""
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create target with class imbalance (5% suspicious)
    y = np.zeros(n_samples)
    n_suspicious = int(n_samples * 0.05)
    suspicious_indices = np.random.choice(n_samples, n_suspicious, replace=False)
    y[suspicious_indices] = 1
    
    # Make suspicious transactions more extreme
    X[suspicious_indices, :10] += 2
    X[suspicious_indices, 10:20] -= 1.5
    
    feature_names = [
        'amount_log', 'amount_zscore', 'hour_of_day', 'day_of_week', 'is_weekend',
        'velocity_7d', 'velocity_30d', 'frequency_7d', 'frequency_30d', 'frequency_change',
        'cross_border', 'high_risk_country', 'round_amount', 'cash_intensive', 'crypto_related',
        'customer_age_days', 'avg_amount_7d', 'avg_amount_30d', 'std_amount_7d', 'std_amount_30d',
        'betweenness_centrality', 'degree_centrality', 'clustering_coef', 'pagerank',
        'time_since_last', 'burst_indicator', 'structuring_indicator', 'layering_score',
        'sanctions_proximity', 'pep_indicator'
    ]
    
    return X, y, feature_names


class MockModel:
    """Mock model for testing when sklearn not available"""
    def __init__(self):
        self.feature_importances_ = np.random.rand(30)
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        # Simple decision boundary
        scores = X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2
        return (scores > 0.5).astype(int)
    
    def predict_proba(self, X):
        pred = self.predict(X)
        return np.column_stack([1-pred, pred])


def demo_attack_generation():
    """Demonstrate various attack generation techniques"""
    print("\n" + "="*80)
    print("DEMO 1: ATTACK GENERATION TECHNIQUES")
    print("="*80)
    
    attack_gen = EvasionAttackGenerator(epsilon=0.1)
    
    # 1. Structuring attacks
    print("\n1️⃣ TRANSACTION STRUCTURING")
    print("-" * 80)
    amounts_to_structure = [15000, 25000, 50000]
    for amount in amounts_to_structure:
        splits = attack_gen.structuring_evasion(amount, threshold=10000)
        print(f"\n💰 Original: ${amount:,.2f}")
        print(f"   Structured into {len(splits)} transactions:")
        for i, split in enumerate(splits, 1):
            print(f"      Transaction {i}: ${split:,.2f}")
    
    # 2. Temporal evasion
    print("\n\n2️⃣ TEMPORAL EVASION")
    print("-" * 80)
    timestamps = np.array([1.0, 1.1, 1.2, 1.3, 5.0, 5.1, 10.0])
    print(f"Original timestamps: {timestamps}")
    adjusted = attack_gen.temporal_evasion(timestamps, velocity_threshold=5.0)
    print(f"Adjusted timestamps: {adjusted}")
    print("✅ Transactions spread out to evade velocity detection")
    
    # 3. Layering attacks
    print("\n\n3️⃣ LAYERING (MONEY FLOW OBFUSCATION)")
    print("-" * 80)
    for num_layers in [2, 5, 10]:
        path = attack_gen.layering_attack("SOURCE_ACCT", "TARGET_ACCT", num_layers)
        print(f"\n{num_layers}-layer path:")
        for i, (src, dst) in enumerate(path, 1):
            print(f"   Step {i}: {src} → {dst}")
    
    # 4. Feature perturbation
    print("\n\n4️⃣ FEATURE PERTURBATION ATTACK")
    print("-" * 80)
    X_sample = np.random.randn(10, 5)
    y_sample = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    
    X_adv, success_mask = attack_gen.feature_perturbation_attack(X_sample, y_sample)
    
    print(f"Generated adversarial examples: {success_mask.sum()}")
    print(f"Average perturbation magnitude: {np.linalg.norm(X_adv - X_sample, axis=1).mean():.4f}")
    print("✅ Features perturbed within realistic constraints")


def demo_robustness_evaluation():
    """Demonstrate model robustness evaluation"""
    print("\n" + "="*80)
    print("DEMO 2: MODEL ROBUSTNESS EVALUATION")
    print("="*80)
    
    # Generate data
    X, y, feature_names = generate_synthetic_aml_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    if SKLEARN_AVAILABLE:
        print("\n🔧 Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        print("✅ Model trained")
    else:
        print("\n🔧 Using Mock Model...")
        model = MockModel()
        model.fit(X_train, y_train)
    
    # Evaluate robustness
    print("\n📊 Evaluating model robustness...")
    evaluator = ModelRobustnessEvaluator(model, feature_names)
    
    # Test multiple attacks
    results = evaluator.comprehensive_evaluation(
        X_test, y_test,
        attack_types=[AttackType.FEATURE_PERTURBATION, AttackType.EVASION]
    )
    
    print("\n" + "-"*80)
    print("ROBUSTNESS RESULTS")
    print("-"*80)
    
    for attack_type, metrics in results.items():
        print(f"\n🎯 {attack_type.upper()}")
        print(f"   Original Accuracy: {metrics.original_accuracy:.4f}")
        print(f"   Adversarial Accuracy: {metrics.adversarial_accuracy:.4f}")
        print(f"   Attack Success Rate: {metrics.attack_success_rate:.2%}")
        print(f"   Robustness Score: {metrics.robustness_score:.4f}")
        print(f"   Detection Rate: {metrics.detection_rate:.2%}")
    
    # Generate report
    print("\n📝 Generating comprehensive report...")
    report = evaluator.generate_report(results)
    
    return model, X_test, y_test, feature_names


def demo_defense_comparison():
    """Compare multiple defense strategies"""
    print("\n" + "="*80)
    print("DEMO 3: DEFENSE STRATEGY COMPARISON")
    print("="*80)
    
    # Generate data
    X, y, feature_names = generate_synthetic_aml_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if not SKLEARN_AVAILABLE:
        print("⚠️  Skipping defense comparison (requires sklearn)")
        return
    
    # Train multiple models with different strategies
    print("\n🔧 Training models with different defense strategies...")
    
    models = {
        'Baseline_RF': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Deep_RF': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
    }
    
    for name, model in models.items():
        print(f"   Training {name}...")
        model.fit(X_train, y_train)
    
    print("✅ All models trained")
    
    # Compare robustness
    print("\n📊 Comparing robustness across models...")
    tester = AdversarialTester(models['Baseline_RF'], feature_names, epsilon=0.1)
    comparison_df = tester.compare_defenses(models, X_test, y_test)
    
    print("\n" + "-"*80)
    print("COMPARISON RESULTS")
    print("-"*80)
    print(comparison_df.to_string(index=False))
    
    # Determine best model
    best_model = comparison_df.loc[comparison_df['avg_robustness'].idxmax(), 'model']
    print(f"\n🏆 Most robust model: {best_model}")


def demo_full_test_suite():
    """Demonstrate complete adversarial test suite"""
    print("\n" + "="*80)
    print("DEMO 4: FULL ADVERSARIAL TEST SUITE")
    print("="*80)
    
    # Generate data
    X, y, feature_names = generate_synthetic_aml_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    if SKLEARN_AVAILABLE:
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
    else:
        model = MockModel()
        model.fit(X_train, y_train)
    
    # Initialize tester
    tester = AdversarialTester(model, feature_names, epsilon=0.1)
    
    # Run full suite
    print("\n🚀 Running full adversarial test suite...")
    results = tester.run_full_test_suite(X_test, y_test, output_dir="adversarial_results")
    
    print("\n✅ Full test suite complete!")
    print(f"\n📊 Summary:")
    print(f"   Attacks tested: {results['summary']['total_attacks_tested']}")
    print(f"   Average robustness: {results['summary']['avg_robustness_score']:.4f}")
    print(f"   Vulnerability level: {results['summary']['vulnerability_level']}")


def demo_specific_scenarios():
    """Test specific real-world AML evasion scenarios"""
    print("\n" + "="*80)
    print("DEMO 5: SPECIFIC EVASION SCENARIOS")
    print("="*80)
    
    # Generate data
    X, y, feature_names = generate_synthetic_aml_data(n_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    if SKLEARN_AVAILABLE:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
    else:
        model = MockModel()
        model.fit(X_train, y_train)
    
    tester = AdversarialTester(model, feature_names, epsilon=0.05)
    
    scenarios = [
        {
            'name': 'Smurfing Attack',
            'config': {
                'type': 'structuring',
                'amounts': [12000, 15000, 20000]
            }
        },
        {
            'name': 'Off-Hours Evasion',
            'config': {
                'type': 'temporal',
                'velocity_threshold': 3.0
            }
        },
        {
            'name': 'Cross-Border Layering',
            'config': {
                'type': 'layering',
                'num_layers': 5
            }
        }
    ]
    
    print("\n🎯 Testing real-world evasion scenarios...\n")
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*80}")
        
        metrics = tester.test_specific_scenario(
            scenario['name'],
            X_test[:50],  # Test on subset
            y_test[:50],
            scenario['config']
        )
        
        print(f"✅ Attack Success Rate: {metrics.attack_success_rate:.2%}")
        print(f"✅ Model Robustness: {metrics.robustness_score:.4f}")


def main():
    """Run all demonstrations"""
    print("\n")
    print("="*80)
    print("🛡️  NEXUS AI - ADVERSARIAL TESTING DEMONSTRATION")
    print("="*80)
    print("\nComprehensive demonstration of adversarial testing capabilities")
    print("for AML/CFT machine learning systems.\n")
    
    try:
        # Run all demos
        demo_attack_generation()
        demo_robustness_evaluation()
        demo_defense_comparison()
        demo_full_test_suite()
        demo_specific_scenarios()
        
        print("\n" + "="*80)
        print("✅ ALL DEMONSTRATIONS COMPLETE")
        print("="*80)
        print("\n📚 Key Takeaways:")
        print("   1. Multiple sophisticated evasion techniques implemented")
        print("   2. Comprehensive robustness evaluation metrics")
        print("   3. Comparison framework for defense strategies")
        print("   4. Production-ready testing infrastructure")
        print("   5. Real-world scenario testing capabilities")
        print("\n🚀 Ready for production deployment!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

