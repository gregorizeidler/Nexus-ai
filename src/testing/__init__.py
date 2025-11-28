"""
NEXUS AI - Adversarial Testing Module

This module provides comprehensive adversarial testing capabilities for AML/CFT systems.
"""

from .adversarial_testing import (
    AdversarialTester,
    EvasionAttackGenerator,
    ModelRobustnessEvaluator,
    AdversarialMetrics
)

__all__ = [
    'AdversarialTester',
    'EvasionAttackGenerator',
    'ModelRobustnessEvaluator',
    'AdversarialMetrics'
]

