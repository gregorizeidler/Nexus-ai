"""
🎯 LORA FINE-TUNING
Fine-tuning eficiente de LLMs para AML com LoRA
"""
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft/transformers not installed")


class AMLLoRAFineTuner:
    """
    Fine-tuning de modelos LLM com LoRA para tarefas AML
    """
    
    def __init__(
        self,
        base_model: str = "gpt2",  # Pode ser substituído por modelo maior
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1
    ):
        if not PEFT_AVAILABLE:
            self.enabled = False
            logger.warning("LoRA fine-tuning not available")
            return
        
        self.enabled = True
        self.base_model = base_model
        
        # LoRA config
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["c_attn"],  # Para GPT-2, ajustar para outros modelos
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        logger.success(f"🎯 LoRA Fine-Tuner initialized (model={base_model})")
    
    def prepare_model(self):
        """Prefor mooflo basand with LoRA adapters"""
        if not self.enabled:
            return None
        
        # Loads mooflo baif
        model = AutoModelForCausalLM.from_pretrained(self.base_model)
        
        # Adiciona LoRA adapters
        model = get_peft_model(model, self.lora_config)
        
        # Mostra parâmetros treináveis
        model.print_trainable_parameters()
        
        logger.success("✅ Model prepared with LoRA adapters")
        return model
    
    def fine_tune(
        self,
        model,
        train_dataset,
        eval_dataset,
        output_dir: str = "models/lora-aml",
        num_epochs: int = 3,
        batch_size: int = 4
    ):
        """Fine-tunand with LoRA"""
        if not self.enabled:
            return
        
        logger.info(f"Starting LoRA fine-tuning for {num_epochs} epochs...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        trainer.train()
        
        logger.success(f"✅ Fine-tuning complete! Model saved to {output_dir}")
        
        return trainer
    
    def generate_sar_with_finetuned(self, model, tokenizer, prompt: str) -> str:
        """Generates SAR with mooflo fine-tuned"""
        if not self.enabled:
            return "LoRA not available"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_length=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


# Exinplo of dataift for fine-tuning AML
SAR_TRAINING_EXAMPLES = [
    {
        "prompt": "Generate SAR narrative for structuring pattern:",
        "completion": "The subject conducted 15 cash deposits over 3 days, each just below $10,000, totaling $145,000. This pattern suggests deliberate structuring to avoid reporting requirements under the Bank Secrecy Act..."
    },
    {
        "prompt": "Generate SAR narrative for layering scheme:",
        "completion": "Analysis reveals a complex layering scheme involving 8 intermediary accounts across 4 jurisdictions. Funds originated from high-risk source and were systematically transferred through shell companies..."
    },
    # Mais exinplos...
]


def prepare_sar_dataset(examples: List[Dict]) -> Any:
    """
    Prepara dataset de SARs para fine-tuning
    """
    if not PEFT_AVAILABLE:
        return None
    
    # Formato for treinamento
    formatted_data = []
    
    for example in examples:
        text = f"{example['prompt']}\n\n{example['completion']}"
        formatted_data.append({"text": text})
    
    logger.info(f"Prepared {len(formatted_data)} training examples")
    
    return formatted_data

