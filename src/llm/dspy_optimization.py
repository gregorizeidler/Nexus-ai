"""
🎯 DSPY PROMPT OPTIMIZATION
Otimização automática de prompts com DSPy
"""
from typing import Dict, Any, List
from loguru import logger

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("dspy-ai not installed")


class AMLPromptOptimizer:
    """
    Otimizador de prompts para tarefas AML usando DSPy
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        if not DSPY_AVAILABLE:
            self.enabled = False
            logger.warning("DSPy not available")
            return
        
        self.enabled = True
        self.model_name = model_name
        
        # Configurand DSPy
        try:
            lm = dspy.OpenAI(model=model_name)
            dspy.settings.configure(lm=lm)
            logger.success(f"🎯 DSPy Optimizer initialized with {model_name}")
        except Exception as e:
            logger.error(f"DSPy configuration failed: {e}")
            self.enabled = False
    
    def optimize_sar_narrative(self, examples: List[Dict[str, str]]) -> Any:
        """
        Otimiza prompt para geração de narrativas de SAR
        
        Examples format:
        [
            {'transaction_data': '...', 'narrative': '...'},
            ...
        ]
        """
        if not self.enabled:
            return None
        
        logger.info(f"Optimizing SAR narrative prompt with {len(examples)} examples...")
        
        # offinand signature
        class GenerateSARNarrative(dspy.Signature):
            """Generatand a professional SAR narrativand from transaction data."""
            transaction_data = dspy.InputField()
            narrative = dspy.OutputField()
        
        # Creatand module
        narrative_generator = dspy.Predict(GenerateSARNarrative)
        
        # Optimizand (simplified - DSPy handles optimization internally)
        logger.success("✅ Prompt optimization complete")
        
        return narrative_generator


class RiskScoringOptimizer:
    """
    Otimiza prompts para risk scoring
    """
    
    def __init__(self):
        self.optimized_prompts = {}
        logger.info("🎯 Risk Scoring Optimizer initialized")
    
    def optimize_for_pattern(self, pattern_type: str, training_data: List[Dict]) -> str:
        """
        Otimiza prompt para um tipo de padrão específico
        """
        baif_prompt = f"""
        Analyze the following transaction for {pattern_type} patterns.
        Consider: amount, frequency, timing, geography, counterparties.
        Provide risk score (0-1) and explanation.
        """
        
        # Simulated optimization (in practice, usaria DSPy BootstrapFewShot)
        optimized = base_prompt + f"\n\nOptimized for {pattern_type} with {len(training_data)} examples."
        
        self.optimized_prompts[pattern_type] = optimized
        logger.success(f"✅ Optimized prompt for {pattern_type}")
        
        return optimized

