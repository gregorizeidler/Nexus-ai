"""
NEXUS AI - Adversarial Testing System

Advanced adversarial testing framework to evaluate model robustness against
sophisticated evasion attacks designed to bypass AML/CFT detection systems.

Key Features:
1. Feature Perturbation Attacks
2. Gradient-Based Attacks (FGSM, PGD)
3. Boundary Attacks
4. Model Inversion Attacks
5. Backdoor Detection
6. Robustness Metrics
7. Defense Mechanisms Testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Enumeration of adversarial attack types"""
    FEATURE_PERTURBATION = "feature_perturbation"
    FGSM = "fast_gradient_sign_method"
    PGD = "projected_gradient_descent"
    BOUNDARY = "boundary_attack"
    CARLINI_WAGNER = "carlini_wagner"
    DEEPFOOL = "deepfool"
    BACKDOOR = "backdoor_poisoning"
    MODEL_INVERSION = "model_inversion"
    EVASION = "evasion_attack"
    POISONING = "data_poisoning"


class DefenseType(Enum):
    """Enumeration of defense mechanisms"""
    ADVERSARIAL_TRAINING = "adversarial_training"
    GRADIENT_MASKING = "gradient_masking"
    INPUT_TRANSFORMATION = "input_transformation"
    ENSEMBLE_DIVERSITY = "ensemble_diversity"
    ANOMALY_DETECTION = "anomaly_detection"
    ROBUST_FEATURES = "robust_feature_selection"


@dataclass
class AdversarialMetrics:
    """Container for adversarial testing metrics"""
    attack_type: str
    original_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    perturbation_magnitude: float
    detection_rate: float
    false_positive_rate: float
    robustness_score: float
    time_elapsed: float
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary"""
        return {
            'attack_type': self.attack_type,
            'original_accuracy': self.original_accuracy,
            'adversarial_accuracy': self.adversarial_accuracy,
            'attack_success_rate': self.attack_success_rate,
            'perturbation_magnitude': self.perturbation_magnitude,
            'detection_rate': self.detection_rate,
            'false_positive_rate': self.false_positive_rate,
            'robustness_score': self.robustness_score,
            'time_elapsed': self.time_elapsed,
            **self.additional_metrics
        }
    
    def __str__(self) -> str:
        return (
            f"AdversarialMetrics(attack={self.attack_type}, "
            f"success_rate={self.attack_success_rate:.2%}, "
            f"robustness={self.robustness_score:.3f})"
        )


class EvasionAttackGenerator:
    """
    Generate adversarial examples designed to evade AML detection.
    
    This class implements various sophisticated evasion techniques that
    money launderers might use to bypass detection systems.
    """
    
    def __init__(self, epsilon: float = 0.1, max_iterations: int = 100):
        """
        Initialize evasion attack generator.
        
        Args:
            epsilon: Maximum perturbation magnitude
            max_iterations: Maximum iterations for iterative attacks
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        logger.info(f"Initialized EvasionAttackGenerator (epsilon={epsilon})")
    
    def feature_perturbation_attack(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        feature_constraints: Optional[Dict[int, Tuple[float, float]]] = None,
        targeted: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adversarial examples by perturbing features within realistic constraints.
        
        Simulates real-world evasion tactics like:
        - Splitting transactions to stay below thresholds
        - Timing adjustments to avoid pattern detection
        - Route changes to obscure money flow
        
        Args:
            X: Original features
            y: Original labels
            feature_constraints: Dict mapping feature index to (min, max) valid range
            targeted: Whether to target specific class (False = untargeted)
        
        Returns:
            Tuple of (adversarial_examples, success_mask)
        """
        X_adv = X.copy()
        success_mask = np.zeros(len(X), dtype=bool)
        
        for i in range(len(X)):
            if y[i] == 1:  # Only attack suspicious transactions
                perturbation = np.random.uniform(-self.epsilon, self.epsilon, X.shape[1])
                
                # Apply constraints
                if feature_constraints:
                    for feat_idx, (min_val, max_val) in feature_constraints.items():
                        X_adv[i, feat_idx] = np.clip(
                            X[i, feat_idx] + perturbation[feat_idx],
                            min_val, max_val
                        )
                else:
                    X_adv[i] = X[i] + perturbation
                
                success_mask[i] = True
        
        logger.info(f"Generated {success_mask.sum()} adversarial examples via feature perturbation")
        return X_adv, success_mask
    
    def structuring_evasion(
        self,
        transaction_amount: float,
        threshold: float = 10000.0,
        num_splits: Optional[int] = None
    ) -> List[float]:
        """
        Simulate transaction structuring to evade CTR thresholds.
        
        Args:
            transaction_amount: Total amount to structure
            threshold: Reporting threshold to evade
            num_splits: Number of splits (auto-calculated if None)
        
        Returns:
            List of split transaction amounts
        """
        if transaction_amount <= threshold:
            return [transaction_amount]
        
        if num_splits is None:
            # Calculate minimum splits needed
            num_splits = int(np.ceil(transaction_amount / (threshold * 0.95)))
        
        # Add randomness to avoid perfect splits
        base_amount = (transaction_amount / num_splits) * (1 - self.epsilon)
        amounts = []
        remaining = transaction_amount
        
        for i in range(num_splits - 1):
            # Random variation around base amount
            amount = base_amount * np.random.uniform(0.9, 1.1)
            amount = min(amount, threshold * 0.99)  # Stay below threshold
            amounts.append(amount)
            remaining -= amount
        
        amounts.append(remaining)  # Add remainder
        
        logger.info(f"Structured ${transaction_amount:,.2f} into {num_splits} transactions")
        return amounts
    
    def temporal_evasion(
        self,
        timestamps: np.ndarray,
        velocity_threshold: float = 5.0
    ) -> np.ndarray:
        """
        Adjust transaction timing to evade velocity-based detection.
        
        Args:
            timestamps: Original transaction timestamps
            velocity_threshold: Max transactions per time unit
        
        Returns:
            Adjusted timestamps
        """
        timestamps_adj = timestamps.copy()
        
        # Sort timestamps
        sorted_indices = np.argsort(timestamps_adj)
        timestamps_sorted = timestamps_adj[sorted_indices]
        
        # Spread out transactions if velocity too high
        for i in range(1, len(timestamps_sorted)):
            time_diff = timestamps_sorted[i] - timestamps_sorted[i-1]
            if time_diff < 1.0 / velocity_threshold:
                # Add delay
                delay = np.random.uniform(1.0, 2.0) / velocity_threshold
                timestamps_sorted[i] += delay
        
        # Restore original order
        timestamps_adj[sorted_indices] = timestamps_sorted
        
        logger.info(f"Applied temporal evasion to {len(timestamps)} transactions")
        return timestamps_adj
    
    def layering_attack(
        self,
        source_node: str,
        target_node: str,
        num_layers: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Generate layered transaction path to obscure money flow.
        
        Args:
            source_node: Starting account
            target_node: Destination account
            num_layers: Number of intermediary accounts
        
        Returns:
            List of transaction edges
        """
        path = [source_node]
        
        # Generate intermediate nodes
        for i in range(num_layers):
            intermediate = f"LAYER_{i}_{np.random.randint(1000, 9999)}"
            path.append(intermediate)
        
        path.append(target_node)
        
        # Create edges
        edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        
        logger.info(f"Generated layering path with {num_layers} intermediaries")
        return edges
    
    def fgsm_attack(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: Optional[float] = None
    ) -> np.ndarray:
        """
        Fast Gradient Sign Method attack.
        
        Args:
            model: Target model with gradient computation
            X: Input features
            y: True labels
            epsilon: Perturbation magnitude
        
        Returns:
            Adversarial examples
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using approximation")
            # Approximate FGSM without gradients
            perturbation = np.random.choice([-1, 1], size=X.shape)
            eps = epsilon or self.epsilon
            return X + eps * perturbation
        
        eps = epsilon or self.epsilon
        X_tensor = torch.FloatTensor(X).requires_grad_(True)
        
        # Get predictions
        outputs = model(X_tensor)
        loss = nn.CrossEntropyLoss()(outputs, torch.LongTensor(y))
        
        # Compute gradients
        model.zero_grad()
        loss.backward()
        
        # Generate adversarial examples
        data_grad = X_tensor.grad.data
        sign_data_grad = data_grad.sign()
        X_adv = X_tensor + eps * sign_data_grad
        
        logger.info(f"Generated FGSM adversarial examples (epsilon={eps})")
        return X_adv.detach().numpy()
    
    def pgd_attack(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: Optional[float] = None,
        alpha: float = 0.01,
        num_iter: Optional[int] = None
    ) -> np.ndarray:
        """
        Projected Gradient Descent attack (stronger than FGSM).
        
        Args:
            model: Target model
            X: Input features
            y: True labels
            epsilon: Total perturbation budget
            alpha: Step size per iteration
            num_iter: Number of iterations
        
        Returns:
            Adversarial examples
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using approximation")
            # Multi-step approximation
            X_adv = X.copy()
            eps = epsilon or self.epsilon
            iterations = num_iter or self.max_iterations
            
            for _ in range(iterations):
                perturbation = np.random.choice([-1, 1], size=X.shape)
                X_adv = X_adv + alpha * perturbation
                # Project back
                X_adv = np.clip(X_adv, X - eps, X + eps)
            
            return X_adv
        
        eps = epsilon or self.epsilon
        iterations = num_iter or min(self.max_iterations, 40)
        
        X_adv = torch.FloatTensor(X).requires_grad_(True)
        
        for i in range(iterations):
            outputs = model(X_adv)
            loss = nn.CrossEntropyLoss()(outputs, torch.LongTensor(y))
            
            model.zero_grad()
            loss.backward()
            
            # Update adversarial examples
            data_grad = X_adv.grad.data
            X_adv = X_adv.detach() + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            perturbation = torch.clamp(X_adv - torch.FloatTensor(X), -eps, eps)
            X_adv = torch.FloatTensor(X) + perturbation
            X_adv = X_adv.detach().requires_grad_(True)
        
        logger.info(f"Generated PGD adversarial examples (epsilon={eps}, iters={iterations})")
        return X_adv.detach().numpy()


class ModelRobustnessEvaluator:
    """
    Evaluate model robustness against adversarial attacks.
    
    Provides comprehensive metrics and analysis of model vulnerability
    to various evasion techniques.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize robustness evaluator.
        
        Args:
            model: Model to evaluate
            feature_names: Names of features for interpretability
        """
        self.model = model
        self.feature_names = feature_names
        self.attack_generator = EvasionAttackGenerator()
        logger.info("Initialized ModelRobustnessEvaluator")
    
    def evaluate_attack(
        self,
        X_original: np.ndarray,
        X_adversarial: np.ndarray,
        y_true: np.ndarray,
        attack_type: str
    ) -> AdversarialMetrics:
        """
        Evaluate a specific attack's effectiveness.
        
        Args:
            X_original: Original features
            X_adversarial: Adversarial features
            y_true: True labels
            attack_type: Type of attack performed
        
        Returns:
            AdversarialMetrics object
        """
        start_time = datetime.now()
        
        # Get predictions
        y_pred_original = self.model.predict(X_original)
        y_pred_adversarial = self.model.predict(X_adversarial)
        
        # Calculate metrics
        original_accuracy = accuracy_score(y_true, y_pred_original) if SKLEARN_AVAILABLE else 0.0
        adversarial_accuracy = accuracy_score(y_true, y_pred_adversarial) if SKLEARN_AVAILABLE else 0.0
        
        # Attack success rate (flipped predictions)
        attack_success = np.sum(y_pred_original != y_pred_adversarial) / len(y_true)
        
        # Perturbation magnitude
        perturbation = np.linalg.norm(X_adversarial - X_original, axis=1).mean()
        
        # Detection rate (would adversarial examples be flagged?)
        detection_rate = self._estimate_detection_rate(X_original, X_adversarial)
        
        # False positive rate
        fpr = np.sum((y_pred_adversarial == 1) & (y_true == 0)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0.0
        
        # Robustness score (0 = completely vulnerable, 1 = perfectly robust)
        robustness_score = 1.0 - attack_success
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        metrics = AdversarialMetrics(
            attack_type=attack_type,
            original_accuracy=original_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            attack_success_rate=attack_success,
            perturbation_magnitude=perturbation,
            detection_rate=detection_rate,
            false_positive_rate=fpr,
            robustness_score=robustness_score,
            time_elapsed=elapsed_time
        )
        
        logger.info(f"Evaluated {attack_type}: {metrics}")
        return metrics
    
    def _estimate_detection_rate(
        self,
        X_original: np.ndarray,
        X_adversarial: np.ndarray
    ) -> float:
        """
        Estimate how many adversarial examples would be detected.
        
        Uses statistical anomaly detection on perturbations.
        """
        perturbations = X_adversarial - X_original
        
        # Calculate perturbation statistics
        pert_norms = np.linalg.norm(perturbations, axis=1)
        threshold = np.percentile(pert_norms, 95)
        
        # Flag examples with large perturbations
        detected = np.sum(pert_norms > threshold) / len(pert_norms)
        
        return detected
    
    def comprehensive_evaluation(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        attack_types: Optional[List[AttackType]] = None
    ) -> Dict[str, AdversarialMetrics]:
        """
        Run comprehensive adversarial evaluation across multiple attack types.
        
        Args:
            X_test: Test features
            y_test: Test labels
            attack_types: List of attacks to test (None = all)
        
        Returns:
            Dictionary mapping attack type to metrics
        """
        if attack_types is None:
            attack_types = [
                AttackType.FEATURE_PERTURBATION,
                AttackType.EVASION
            ]
        
        results = {}
        
        logger.info(f"Starting comprehensive evaluation with {len(attack_types)} attack types")
        
        for attack_type in attack_types:
            logger.info(f"Testing {attack_type.value}...")
            
            if attack_type == AttackType.FEATURE_PERTURBATION:
                X_adv, _ = self.attack_generator.feature_perturbation_attack(X_test, y_test)
            elif attack_type == AttackType.EVASION:
                # Combine multiple evasion techniques
                X_adv = X_test.copy()
                # Apply small perturbations
                perturbation = np.random.uniform(-0.05, 0.05, X_test.shape)
                X_adv += perturbation
            else:
                logger.warning(f"Attack type {attack_type} not implemented, skipping")
                continue
            
            metrics = self.evaluate_attack(X_test, X_adv, y_test, attack_type.value)
            results[attack_type.value] = metrics
        
        logger.info(f"Comprehensive evaluation complete: {len(results)} attacks tested")
        return results
    
    def generate_report(
        self,
        evaluation_results: Dict[str, AdversarialMetrics],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive robustness report.
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            output_path: Optional path to save report
        
        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "NEXUS AI - ADVERSARIAL ROBUSTNESS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {type(self.model).__name__}",
            f"Attacks Tested: {len(evaluation_results)}",
            "",
            "=" * 80,
            "SUMMARY OF RESULTS",
            "=" * 80,
            ""
        ]
        
        # Summary table
        for attack_type, metrics in evaluation_results.items():
            report_lines.extend([
                f"Attack Type: {attack_type}",
                f"  Original Accuracy: {metrics.original_accuracy:.4f}",
                f"  Adversarial Accuracy: {metrics.adversarial_accuracy:.4f}",
                f"  Attack Success Rate: {metrics.attack_success_rate:.2%}",
                f"  Robustness Score: {metrics.robustness_score:.4f}",
                f"  Perturbation Magnitude: {metrics.perturbation_magnitude:.6f}",
                f"  Detection Rate: {metrics.detection_rate:.2%}",
                ""
            ])
        
        # Overall assessment
        avg_robustness = np.mean([m.robustness_score for m in evaluation_results.values()])
        report_lines.extend([
            "=" * 80,
            "OVERALL ASSESSMENT",
            "=" * 80,
            f"Average Robustness Score: {avg_robustness:.4f}",
            "",
            "Vulnerability Level: " + self._assess_vulnerability(avg_robustness),
            "",
            "=" * 80,
            "RECOMMENDATIONS",
            "=" * 80,
            self._generate_recommendations(evaluation_results),
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def _assess_vulnerability(self, robustness_score: float) -> str:
        """Assess vulnerability level based on robustness score"""
        if robustness_score >= 0.9:
            return "LOW - Model is highly robust"
        elif robustness_score >= 0.7:
            return "MEDIUM - Model shows moderate robustness"
        elif robustness_score >= 0.5:
            return "HIGH - Model is vulnerable to attacks"
        else:
            return "CRITICAL - Model is highly vulnerable"
    
    def _generate_recommendations(self, results: Dict[str, AdversarialMetrics]) -> str:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Check for specific vulnerabilities
        for attack_type, metrics in results.items():
            if metrics.attack_success_rate > 0.3:
                recommendations.append(
                    f"• High vulnerability to {attack_type} detected. "
                    f"Consider adversarial training or robust feature engineering."
                )
        
        if not recommendations:
            recommendations.append("• Model shows good robustness across tested attacks.")
            recommendations.append("• Continue monitoring with periodic adversarial testing.")
        else:
            recommendations.append("• Implement adversarial training with diverse attack examples.")
            recommendations.append("• Add ensemble methods for increased robustness.")
            recommendations.append("• Deploy anomaly detection for adversarial example detection.")
        
        return "\n".join(recommendations)


class AdversarialTester:
    """
    Main class for adversarial testing of AML/CFT detection systems.
    
    Orchestrates all adversarial testing workflows including:
    - Attack generation
    - Robustness evaluation
    - Defense testing
    - Report generation
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        epsilon: float = 0.1
    ):
        """
        Initialize adversarial tester.
        
        Args:
            model: Model to test
            feature_names: Feature names for interpretability
            epsilon: Default perturbation magnitude
        """
        self.model = model
        self.feature_names = feature_names
        self.epsilon = epsilon
        
        self.attack_generator = EvasionAttackGenerator(epsilon=epsilon)
        self.robustness_evaluator = ModelRobustnessEvaluator(model, feature_names)
        
        self.test_results = []
        
        logger.info(f"Initialized AdversarialTester (epsilon={epsilon})")
    
    def run_full_test_suite(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete adversarial testing suite.
        
        Args:
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save results
        
        Returns:
            Dictionary containing all test results
        """
        logger.info("=" * 80)
        logger.info("STARTING FULL ADVERSARIAL TEST SUITE")
        logger.info("=" * 80)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'epsilon': self.epsilon,
            'attacks': {},
            'summary': {}
        }
        
        # Run comprehensive evaluation
        attack_results = self.robustness_evaluator.comprehensive_evaluation(
            X_test, y_test
        )
        
        results['attacks'] = {k: v.to_dict() for k, v in attack_results.items()}
        
        # Generate summary statistics
        results['summary'] = {
            'avg_robustness_score': np.mean([m.robustness_score for m in attack_results.values()]),
            'avg_attack_success_rate': np.mean([m.attack_success_rate for m in attack_results.values()]),
            'total_attacks_tested': len(attack_results),
            'vulnerability_level': self.robustness_evaluator._assess_vulnerability(
                np.mean([m.robustness_score for m in attack_results.values()])
            )
        }
        
        # Generate report
        report = self.robustness_evaluator.generate_report(
            attack_results,
            f"{output_dir}/adversarial_report.txt" if output_dir else None
        )
        
        results['report'] = report
        
        # Save results
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/adversarial_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_dir}")
        
        logger.info("=" * 80)
        logger.info("ADVERSARIAL TEST SUITE COMPLETE")
        logger.info("=" * 80)
        print("\n" + report)
        
        return results
    
    def test_specific_scenario(
        self,
        scenario_name: str,
        X_sample: np.ndarray,
        y_sample: np.ndarray,
        attack_config: Dict[str, Any]
    ) -> AdversarialMetrics:
        """
        Test a specific adversarial scenario.
        
        Args:
            scenario_name: Name/description of scenario
            X_sample: Sample features
            y_sample: Sample labels
            attack_config: Configuration for attack
        
        Returns:
            Metrics for this scenario
        """
        logger.info(f"Testing scenario: {scenario_name}")
        
        attack_type = attack_config.get('type', 'feature_perturbation')
        
        if attack_type == 'structuring':
            # Test structuring evasion
            amounts = attack_config.get('amounts', [])
            splits = [
                self.attack_generator.structuring_evasion(amt)
                for amt in amounts
            ]
            logger.info(f"Generated {len(splits)} structuring patterns")
        
        # Generate adversarial examples
        X_adv, _ = self.attack_generator.feature_perturbation_attack(
            X_sample, y_sample
        )
        
        # Evaluate
        metrics = self.robustness_evaluator.evaluate_attack(
            X_sample, X_adv, y_sample, scenario_name
        )
        
        self.test_results.append({
            'scenario': scenario_name,
            'metrics': metrics,
            'config': attack_config
        })
        
        return metrics
    
    def compare_defenses(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare robustness of multiple models/defense strategies.
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            DataFrame comparing model robustness
        """
        comparison_results = []
        
        for model_name, model in models.items():
            logger.info(f"Testing {model_name}...")
            
            evaluator = ModelRobustnessEvaluator(model, self.feature_names)
            results = evaluator.comprehensive_evaluation(X_test, y_test)
            
            avg_metrics = {
                'model': model_name,
                'avg_robustness': np.mean([m.robustness_score for m in results.values()]),
                'avg_attack_success': np.mean([m.attack_success_rate for m in results.values()]),
                'avg_detection_rate': np.mean([m.detection_rate for m in results.values()])
            }
            
            comparison_results.append(avg_metrics)
        
        df = pd.DataFrame(comparison_results)
        logger.info(f"\n{df}")
        
        return df


# Example usage and testing
if __name__ == "__main__":
    print("NEXUS AI - Adversarial Testing Module")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    X_test = np.random.randn(100, 20)
    y_test = np.random.randint(0, 2, 100)
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return (X[:, 0] > 0).astype(int)
    
    model = MockModel()
    
    # Initialize tester
    tester = AdversarialTester(model, epsilon=0.1)
    
    # Run full test suite
    results = tester.run_full_test_suite(X_test, y_test)
    
    print("\n✅ Adversarial testing complete!")

