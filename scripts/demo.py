"""
Demo script to showcase the Titans architecture with a simple example.
"""

import argparse
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import get_model
from configs.model_configs import get_config
from utils.tokenizer import TokenizerWrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demo for Titans architecture")
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="mac",
        choices=["mac", "mag", "mal", "lmm"],
        help="Model type (Memory as Context, Memory as Gate, Memory as Layer, or Long-term Memory Module)"
    )
    
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large"],
        help="Model size"
    )
    
    parser.add_argument(
        "--input_text",
        type=str,
        default="The Titans architecture introduces three levels of memory:",
        help="Input text for the model"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum length of generated text"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu)"
    )
    
    return parser.parse_args()


def main():
    """Main function for the demo script."""
    args = parse_args()
    
    print(f"Running Titans demo with model type: {args.model_type}, size: {args.model_size}")
    print(f"Using device: {args.device}")
    
    # Initialize tokenizer
    tokenizer = TokenizerWrapper("gpt2")
    print("Tokenizer initialized")
    
    # Get model configuration
    config = get_config(model_size=args.model_size, model_type=args.model_type)
    config_dict = config.to_dict()
    
    # Update configuration with actual vocabulary size
    config_dict["vocab_size"] = tokenizer.vocab_size
    
    # Initialize model
    print("Initializing model...")
    model = get_model(config.model_type, **config_dict)
    model.to(args.device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Encode input text
    input_ids = tokenizer.encode(
        args.input_text,
        return_tensors="pt"
    ).input_ids.to(args.device)
    
    print(f"Input text: {args.input_text}")
    
    # Generate text
    print(f"Generating text with temperature {args.temperature}...")
    
    if args.model_type == "lmm" and hasattr(model, "generate"):
        # LMM model has a specialized generate method
        output_ids, _ = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature
        )
    else:
        # For other models, implement manual generation
        output_ids = input_ids
        memory = None
        
        for _ in range(args.max_length - input_ids.shape[1]):
            # Forward pass
            with torch.no_grad():
                logits, memory = model(output_ids, memory)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / args.temperature
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with the generated sequence
            output_ids = torch.cat([output_ids, next_token], dim=1)
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0])
    print("\nGenerated text:")
    print(generated_text)


if __name__ == "__main__":
    main() 