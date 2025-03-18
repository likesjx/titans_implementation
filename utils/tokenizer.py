"""
Tokenization utilities for the Titans implementation.
"""

import torch
from transformers import AutoTokenizer


class TokenizerWrapper:
    """
    Wrapper around Hugging Face tokenizers for use with Titans models.
    """
    
    def __init__(self, tokenizer_name="gpt2"):
        """
        Initialize the tokenizer.
        
        Args:
            tokenizer_name: Name of the Hugging Face tokenizer to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Set padding token to EOS token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def encode(self, text, max_length=None, padding=True, truncation=True, return_tensors="pt"):
        """
        Encode text into token IDs.
        
        Args:
            text: Text or list of texts to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Return format ('pt' for PyTorch tensors)
            
        Returns:
            Encoded token IDs
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the decoded text
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
            
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(self, token_ids_batch, skip_special_tokens=True):
        """
        Decode a batch of token IDs back to text.
        
        Args:
            token_ids_batch: Batch of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the decoded text
            
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
            
        return self.tokenizer.batch_decode(token_ids_batch, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self):
        """Get the vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def eos_token_id(self):
        """Get the end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def pad_token_id(self):
        """Get the padding token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def bos_token_id(self):
        """Get the beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id 