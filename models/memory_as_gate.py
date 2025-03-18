import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import TitansBaseModel
from .long_term_memory import ParallelLongTermMemory


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head attention and feed-forward network.
    """
    
    def __init__(self, hidden_size, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        """
        Forward pass through the encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            attn_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(
            query=self.norm1(x),
            key=self.norm1(x),
            value=self.norm1(x),
            attn_mask=attn_mask
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class MemoryGateLayer(nn.Module):
    """
    Memory gate layer that applies a gating mechanism based on memory output.
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        
        self.gate_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x, memory_output):
        """
        Apply memory-based gating to the input.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            memory_output: Memory output [batch_size, seq_len, hidden_size]
            
        Returns:
            Gated output tensor [batch_size, seq_len, hidden_size]
        """
        # Compute gate values
        gate = torch.sigmoid(self.gate_linear(memory_output))
        
        # Apply gating
        output = x * gate
        
        return output


class MemoryAsGateModel(TitansBaseModel):
    """
    Implementation of the Memory as a Gate (MAG) approach from the Titans paper.
    
    In this approach, only persistent memory is added to the context, and
    lookup results from long-term memory are used to control the output
    via a gating mechanism.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        ff_dim,
        max_seq_len,
        window_size=128,
        memory_hidden_dims=[256, 128],
        persistent_memory_size=16,
        dropout=0.1,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            persistent_memory_size=persistent_memory_size,
            dropout=dropout,
        )
        
        # Sliding window size for attention
        self.window_size = window_size
        
        # Long-term memory module
        self.long_term_memory = ParallelLongTermMemory(
            input_dim=hidden_size,
            hidden_dims=memory_hidden_dims,
            output_dim=hidden_size,
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Memory gate layers
        self.memory_gates = nn.ModuleList([
            MemoryGateLayer(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        
    def init_memory(self, batch_size, device):
        """
        Initialize long-term memory.
        
        Args:
            batch_size: Number of sequences in the batch
            device: Device to initialize tensors on
            
        Returns:
            Initialized memory state
        """
        return self.long_term_memory.memory_mlp.initialize(batch_size, device)
    
    def update_memory(self, memory, keys, values):
        """
        Update long-term memory.
        
        Args:
            memory: Current memory state
            keys: Input keys
            values: Target values
            
        Returns:
            Updated memory state
        """
        _, updated_memory = self.long_term_memory.process_batch(
            keys=keys,
            values=values,
            memory_state=memory
        )
        return updated_memory
    
    def retrieve_from_memory(self, memory, query):
        """
        Retrieve values from long-term memory.
        
        Args:
            memory: Current memory state
            query: Query tensor
            
        Returns:
            Retrieved values
        """
        retrieved_values, _ = self.long_term_memory.process_batch(
            keys=query,
            values=query,  # Dummy values, not used for retrieval
            memory_state=memory
        )
        return retrieved_values
    
    def forward(self, inputs, memory=None):
        """
        Forward pass through the Memory as Gate (MAG) model.
        
        Args:
            inputs: Input token ids [batch_size, seq_len]
            memory: Optional memory state from previous steps
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            new_memory: Updated memory state
        """
        batch_size, seq_len = inputs.shape
        device = inputs.device
        
        # Initialize memory if not provided
        if memory is None:
            memory = self.init_memory(batch_size, device)
        
        # Prepare input embeddings
        embeddings, persistent_memory = self.prepare_inputs(inputs)
        
        # Concatenate persistent memory with input
        # [batch_size, persistent_size + seq_len, hidden_size]
        extended_input = torch.cat([persistent_memory, embeddings], dim=1)
        
        # Create sliding window attention mask
        attn_mask = self._create_sliding_window_mask(
            persistent_size=self.persistent_memory_size,
            seq_len=seq_len,
            window_size=self.window_size
        ).to(device)
        
        # Pass through transformer layers with memory gating
        hidden_states = extended_input
        current_memory = memory
        
        for i, (encoder_layer, gate_layer) in enumerate(zip(self.encoder_layers, self.memory_gates)):
            # Process through transformer layer
            layer_output = encoder_layer(hidden_states, attn_mask)
            
            # Extract the sequence part (excluding persistent memory)
            seq_output = layer_output[:, self.persistent_memory_size:, :]
            
            # Retrieve from memory
            memory_output = self.retrieve_from_memory(
                memory=current_memory,
                query=seq_output
            )
            
            # Apply memory gating to sequence output
            gated_output = gate_layer(seq_output, memory_output)
            
            # Update the hidden states with gated output
            hidden_states = torch.cat([
                layer_output[:, :self.persistent_memory_size, :],  # Persistent memory part
                gated_output  # Gated sequence part
            ], dim=1)
            
            # Update memory with current layer's output
            if i == len(self.encoder_layers) - 1:  # Only update on last layer
                current_memory = self.update_memory(
                    memory=current_memory,
                    keys=seq_output,
                    values=seq_output
                )
        
        # Extract final output (excluding persistent memory)
        hidden_states = hidden_states[:, self.persistent_memory_size:, :]
        
        # Apply final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary
        logits = self.output_layer(hidden_states)
        
        return logits, current_memory
    
    def _create_sliding_window_mask(self, persistent_size, seq_len, window_size):
        """
        Create sliding window attention mask.
        
        This mask allows:
        - Full attention within persistent memory
        - Local attention within a sliding window for the sequence
        - Full attention from sequence to persistent memory
        
        Args:
            persistent_size: Size of persistent memory
            seq_len: Length of input sequence
            window_size: Size of sliding window
            
        Returns:
            Attention mask tensor
        """
        total_size = persistent_size + seq_len
        
        # Start with fully masked tensor (no attention allowed)
        mask = torch.ones(total_size, total_size, dtype=torch.bool)
        
        # Allow attention within persistent memory
        mask[:persistent_size, :persistent_size] = False
        
        # Allow attention within sliding window for sequence
        for i in range(seq_len):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_len, i + window_size // 2 + 1)
            
            # Allow attention within the window
            mask_start = persistent_size + i
            for j in range(window_start, window_end):
                mask_j = persistent_size + j
                mask[mask_start, mask_j] = False
        
        # Allow sequence to attend to persistent memory
        for i in range(seq_len):
            mask_i = persistent_size + i
            mask[mask_i, :persistent_size] = False
        
        return mask 