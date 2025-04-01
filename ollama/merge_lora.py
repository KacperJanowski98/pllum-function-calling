"""
This script merges LoRA adapters with the base model to create a full model.

Usage:
  python merge_lora.py --input-dir models/pllum-function-calling-20250330_071532 --output-dir models/pllum-function-calling-merged
"""

import os
import argparse
import torch
from peft import AutoPeftModelForCausalLM  # Changed from AutoPeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_lora_model(input_dir, output_dir, base_model_id=None):
    """
    Merge LoRA adapters with the base model to create a full model.
    
    Args:
        input_dir: Path to the fine-tuned model with LoRA adapters
        output_dir: Path to save the merged model
        base_model_id: Optional base model ID (if not provided, will be inferred)
    """
    logger.info(f"Loading fine-tuned model from {input_dir}")
    
    # Use BitsAndBytes config for loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load the fine-tuned model with LoRA adapters
    # Changed from AutoPeftModel to AutoPeftModelForCausalLM
    model = AutoPeftModelForCausalLM.from_pretrained(
        input_dir,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    logger.info("Model loaded. Merging adapters with base model...")
    
    # Merge adapters with the base model
    merged_model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving merged model to {output_dir}")
    
    # Save the merged model
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB"
    )
    
    # Save the tokenizer
    logger.info("Saving tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model and tokenizer saved successfully!")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument("--input-dir", required=True, help="Path to fine-tuned model with LoRA adapters")
    parser.add_argument("--output-dir", required=True, help="Path to save merged model")
    parser.add_argument("--base-model-id", help="Base model ID (optional)")
    
    args = parser.parse_args()
    merge_lora_model(args.input_dir, args.output_dir, args.base_model_id)
