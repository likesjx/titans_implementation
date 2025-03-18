"""
Data handling utilities for the Titans implementation.
"""

import os
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


def create_sample_data(output_dir):
    """Create sample data for training and testing."""
    print(f"Creating sample data in {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create a simple dataset for demonstration
    train_data = {
        "sequences": [
            {"text": "This is a sample sequence for short-term memory testing."},
            {"text": "Titans architecture uses three levels of memory mechanisms."},
            {"text": "The long-term memory module updates during inference."},
            {"text": "Test-time learning is a key innovation in Titans."},
            {"text": "Memory as a Context integrates retrieved information."},
            {"text": "Memory as a Gate controls the output via a gating mechanism."},
            {"text": "The persistent memory component stores task-specific information."},
            {"text": "Each memory mechanism serves a unique purpose in the architecture."}
        ]
    }
    
    # Save the sample data
    with open(os.path.join(output_dir, "train_sample.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    
    # Create validation data
    val_data = {
        "sequences": [
            {"text": "The model combines different types of memory for better performance."},
            {"text": "Short-term memory is implemented using transformer attention."},
            {"text": "Long-term memory learns to store and retrieve information."},
            {"text": "The architecture is designed for efficient inference and adaptation."}
        ]
    }
    
    with open(os.path.join(output_dir, "val_sample.json"), "w") as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Sample data created successfully in {output_dir}")


class TitansDataset(Dataset):
    """Dataset for Titans model training and evaluation."""
    
    def __init__(self, data_path, tokenizer, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (.json)
            tokenizer: TokenizerWrapper instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.sequences = self.data["sequences"]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = self.sequences[idx]
        text = item["text"]
        
        # Tokenize text
        encoding = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract tensors
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }


def get_dataloader(data_path, tokenizer, batch_size=32, max_length=128, shuffle=True):
    """Create a DataLoader for the dataset."""
    dataset = TitansDataset(data_path, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 