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


class MemoryAsLayerModel(TitansBaseModel):
    """
    Implementation of the Memory as a Layer (MAL) approach from the Titans paper.
    
    In this approach, the memory layer serves as a preprocessing mechanism
    before the context (extended with persistent memory) is fed into the
    attention mechanism.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        ff_dim,
        max_seq_len,
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
        
        # Memory preprocessing layer
        self.memory_projection = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for memory output
        self.memory_layer_norm = nn.LayerNorm(hidden_size)
        
        # Final layer normalization
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
        Forward pass through the Memory as Layer (MAL) model.
        
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
        
        # First, process through memory layer as preprocessing
        memory_output = self.retrieve_from_memory(
            memory=memory,
            query=embeddings
        )
        
        # Project and normalize memory output
        memory_output = self.memory_projection(memory_output)
        memory_output = self.memory_layer_norm(memory_output)
        
        # Add memory output to embeddings (residual connection)
        preprocessed_input = embeddings + memory_output
        
        # Concatenate persistent memory with preprocessed input
        # [batch_size, persistent_size + seq_len, hidden_size]
        extended_input = torch.cat([persistent_memory, preprocessed_input], dim=1)
        
        # Create causal attention mask
        attn_mask = self._create_causal_attention_mask(
            persistent_size=self.persistent_memory_size,
            seq_len=seq_len
        ).to(device)
        
        # Pass through transformer layers
        hidden_states = extended_input
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attn_mask)
        
        # Extract outputs (excluding persistent memory)
        hidden_states = hidden_states[:, self.persistent_memory_size:, :]
        
        # Apply final layer norm and dropout
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary
        logits = self.output_layer(hidden_states)
        
        # Update memory with current sequence
        updated_memory = self.update_memory(
            memory=memory,
            keys=embeddings,
            values=embeddings
        )
        
        return logits, updated_memory
    
    def _create_causal_attention_mask(self, persistent_size, seq_len):
        """
        Create causal attention mask that allows:
        - Full attention within persistent memory
        - Causal attention within the sequence
        - Full attention from sequence to persistent memory
        
        Args:
            persistent_size: Size of persistent memory
            seq_len: Length of input sequence
            
        Returns:
            Attention mask tensor
        """
        total_size = persistent_size + seq_len
        
        # Start with fully masked tensor (no attention allowed)
        mask = torch.ones(total_size, total_size, dtype=torch.bool)
        
        # Allow attention within persistent memory
        mask[:persistent_size, :persistent_size] = False
        
        # Allow causal attention within sequence
        for i in range(seq_len):
            # Position in the extended sequence
            pos_i = persistent_size + i
            
            # Allow attention to previous positions and self
            for j in range(i + 1):
                pos_j = persistent_size + j
                mask[pos_i, pos_j] = False
        
        # Allow sequence to attend to persistent memory
        for i in range(seq_len):
            pos_i = persistent_size + i
            mask[pos_i, :persistent_size] = False
        
        return mask 