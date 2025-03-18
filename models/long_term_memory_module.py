import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import TitansBaseModel
from .long_term_memory import ParallelLongTermMemory


class LongTermMemoryModule(TitansBaseModel):
    """
    Implementation of the Long-term Memory Module (LMM) approach from the Titans paper.
    
    This is a purely recurrent approach that does without self-attention entirely,
    using only the long-term memory mechanism as a processing layer.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        ff_dim,
        max_seq_len,
        memory_hidden_dims=[256, 128],
        persistent_memory_size=16,
        dropout=0.1,
    ):
        # Note: num_heads is not used in this model but kept for compatibility
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=1,  # Dummy value
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            persistent_memory_size=persistent_memory_size,
            dropout=dropout,
        )
        
        # Long-term memory modules (one for each layer)
        self.memory_layers = nn.ModuleList([
            ParallelLongTermMemory(
                input_dim=hidden_size,
                hidden_dims=memory_hidden_dims,
                output_dim=hidden_size,
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward layers
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, hidden_size)
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        
        # Persistent memory projection
        self.persistent_proj = nn.Linear(hidden_size, hidden_size)
        
    def init_memory(self, batch_size, device):
        """
        Initialize long-term memory for all layers.
        
        Args:
            batch_size: Number of sequences in the batch
            device: Device to initialize tensors on
            
        Returns:
            List of initialized memory states
        """
        return [
            memory_layer.memory_mlp.initialize(batch_size, device)
            for memory_layer in self.memory_layers
        ]
    
    def forward(self, inputs, memory=None):
        """
        Forward pass through the Long-term Memory Module (LMM) model.
        
        Args:
            inputs: Input token ids [batch_size, seq_len]
            memory: Optional list of memory states from previous steps
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            new_memory: Updated list of memory states
        """
        batch_size, seq_len = inputs.shape
        device = inputs.device
        
        # Initialize memory if not provided
        if memory is None:
            memory = self.init_memory(batch_size, device)
        
        # Prepare input embeddings
        embeddings, persistent_memory_tokens = self.prepare_inputs(inputs)
        
        # Project persistent memory
        persistent_memory = self.persistent_proj(persistent_memory_tokens)
        
        # Process through memory layers
        hidden_states = embeddings
        new_memory = []
        
        for i, (memory_layer, ff_layer, norm1, norm2) in enumerate(
            zip(self.memory_layers, self.feed_forward_layers, self.layer_norms_1, self.layer_norms_2)
        ):
            # Apply persistent memory attention (simple addition as a form of bias)
            # This simulates attending to persistent memory without using attention
            persistent_bias = persistent_memory.mean(dim=1, keepdim=True)
            hidden_states = hidden_states + persistent_bias
            
            # Retrieve from memory and update with residual connection
            memory_output, updated_memory = memory_layer.process_batch(
                keys=hidden_states,
                values=hidden_states,
                memory_state=memory[i]
            )
            
            # Store updated memory
            new_memory.append(updated_memory)
            
            # Add residual connection and normalize
            hidden_states = norm1(hidden_states + memory_output)
            
            # Apply feed-forward layer with residual connection
            ff_output = ff_layer(hidden_states)
            hidden_states = norm2(hidden_states + ff_output)
        
        # Apply final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary
        logits = self.output_layer(hidden_states)
        
        return logits, new_memory
    
    def generate(self, input_ids, max_length, memory=None, temperature=1.0):
        """
        Generate text using the model in an autoregressive manner.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            max_length: Maximum length of the generated sequence
            memory: Optional list of memory states from previous steps
            temperature: Sampling temperature
            
        Returns:
            generated_ids: Generated token ids [batch_size, max_length]
            final_memory: Final memory state
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize memory if not provided
        if memory is None:
            memory = self.init_memory(batch_size, device)
        
        # Initialize generated ids with input ids
        generated_ids = input_ids
        
        # Generate tokens autoregressively
        current_memory = memory
        for _ in range(max_length - input_ids.shape[1]):
            # Get predictions for the next token
            logits, current_memory = self.forward(generated_ids, current_memory)
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate with the generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Break if all sequences have reached the end token (optional)
            # if (next_token == eos_token_id).all():
            #     break
        
        return generated_ids, current_memory 