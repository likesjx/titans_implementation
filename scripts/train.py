"""
Training script for the Titans architecture.
"""

import argparse
import torch
import sys
import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.model_configs import get_config
from models import get_model
from utils.data import create_sample_data, TitansDataset
from utils.tokenizer import TokenizerWrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training for Titans architecture")
    parser.add_argument("--model_size", type=str, default="small",
                        help="Model size: tiny, small, medium, or large")
    parser.add_argument("--model_type", type=str, default="mac",
                        help="Model type: mac, mag, mal, or lmm")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation data")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model outputs")
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create sample data for testing")
    parser.add_argument("--sample_data_dir", type=str, default="data/sample",
                        help="Directory to store sample data")
    return parser.parse_args()


def create_dataloaders(config, train_path, val_path, tokenizer):
    """Create training and validation dataloaders."""
    train_dataset = TitansDataset(
        data_path=train_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )
    
    val_dataset = TitansDataset(
        data_path=val_path,
        tokenizer=tokenizer,
        max_length=config.max_seq_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, scheduler, device, memories=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    # Initialize memories if not provided
    if memories is None:
        memories = model.init_memory(dataloader.batch_size, device)
    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch in pbar:
            # Get batch data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift input_ids and create target labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]  # Shift left by 1
            labels[:, -1] = model.tokenizer.pad_token_id  # Pad last position
            
            # Forward pass
            logits, new_memories = model(
                inputs=input_ids,
                memory=memories
            )
            
            # Calculate loss (ignore padding tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model.vocab_size),
                labels.view(-1),
                ignore_index=model.tokenizer.pad_token_id
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update memories
            memories = new_memories
            
            # Update progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
    
    return total_loss / len(dataloader), memories


def validate(model, dataloader, device, memories=None):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    # Initialize memories if not provided
    if memories is None:
        memories = model.init_memory(dataloader.batch_size, device)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Shift input_ids and create target labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]  # Shift left by 1
            labels[:, -1] = model.tokenizer.pad_token_id  # Pad last position
            
            # Forward pass
            logits, new_memories = model(
                inputs=input_ids,
                memory=memories
            )
            
            # Calculate loss (ignore padding tokens)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model.vocab_size),
                labels.view(-1),
                ignore_index=model.tokenizer.pad_token_id
            )
            
            total_loss += loss.item()
            memories = new_memories
    
    return total_loss / len(dataloader), memories


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save a training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)


def main():
    """Main function for training."""
    args = parse_args()
    
    if args.create_sample_data:
        create_sample_data(args.sample_data_dir)
        return
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config(args.model_size, args.model_type)
    
    # Update config with command line arguments
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Initialize tokenizer
    tokenizer = TokenizerWrapper()
    config.vocab_size = tokenizer.vocab_size  # Update vocab size to match tokenizer
    
    # Create model using the factory function
    model = get_model(
        model_type=config.model_type,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        max_seq_len=config.max_seq_len,
        persistent_memory_size=config.persistent_memory_size,
        dropout=config.dropout,
        # Model specific parameters
        chunk_size=config.chunk_size if config.model_type == "mac" else None,
        window_size=config.window_size if config.model_type == "mag" else None,
        memory_hidden_dims=config.memory_hidden_dims,
        tokenizer=tokenizer  # Pass tokenizer to model
    )
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        config,
        args.train_data,
        args.val_data,
        tokenizer
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader)
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize memories
    train_memories = model.init_memory(config.batch_size, device)
    val_memories = model.init_memory(config.batch_size, device)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training phase
        train_loss, train_memories = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            train_memories
        )
        
        # Validation phase
        val_loss, val_memories = validate(
            model,
            val_loader,
            device,
            val_memories
        )
        
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                output_dir / 'best_model.pt'
            )
        
        # Save latest checkpoint
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            val_loss,
            output_dir / 'latest_model.pt'
        )


if __name__ == "__main__":
    main() 