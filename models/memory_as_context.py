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


class MemoryAsContextModel(TitansBaseModel):
    """
    Implementation of the Memory as a Context (MAC) approach from the Titans paper.
    
    In this approach, memory retrieval results are concatenated with the input sequence
    before being processed by the transformer layers.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers,
        num_heads,
        ff_dim,
        max_seq_len,
        chunk_size=64,
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
        
        # Chunk size for processing sequences
        self.chunk_size = chunk_size
        
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
        
        # Embedding layer to convert memory output for concatenation
        self.memory_proj = nn.Linear(hidden_size, hidden_size)
        
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
        Forward pass through the Memory as Context (MAC) model.
        
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
        
        # Process sequence in chunks
        chunk_outputs = []
        current_memory = memory
        
        for i in range(0, seq_len, self.chunk_size):
            # Get current chunk
            chunk_end = min(i + self.chunk_size, seq_len)
            chunk = embeddings[:, i:chunk_end, :]  # [batch_size, chunk_size, hidden_size]
            
            # Retrieve from memory
            memory_values = self.retrieve_from_memory(
                memory=current_memory,
                query=chunk
            )
            
            # Project memory values
            memory_values = self.memory_proj(memory_values)
            
            # Concatenate persistent memory and memory values with input
            # [batch_size, persistent_size + chunk_size, hidden_size]
            extended_input = torch.cat([persistent_memory, memory_values, chunk], dim=1)
            
            # Create attention mask to allow self-attention only within each component
            attn_mask = self._create_attention_mask(
                persistent_size=self.persistent_memory_size,
                memory_size=chunk_end - i,
                chunk_size=chunk_end - i
            ).to(device)
            
            # Pass through transformer layers
            hidden_states = extended_input
            for layer in self.encoder_layers:
                hidden_states = layer(hidden_states, attn_mask)
            
            # Extract chunk outputs (excluding persistent memory and memory values)
            chunk_output = hidden_states[:, self.persistent_memory_size + (chunk_end - i):, :]
            chunk_outputs.append(chunk_output)
            
            # Update memory with current chunk
            current_memory = self.update_memory(
                memory=current_memory,
                keys=chunk,
                values=chunk
            )
        
        # Concatenate chunk outputs
        hidden_states = torch.cat(chunk_outputs, dim=1)
        
        # Apply final layer norm and dropout
        hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary
        logits = self.output_layer(hidden_states)
        
        return logits, current_memory
    
    def _create_attention_mask(self, persistent_size, memory_size, chunk_size):
        """
        Create attention mask for the extended input.
        
        This mask allows:
        - Full attention within persistent memory
        - Full attention within memory values
        - Full attention within input chunk
        - Memory values and input chunk can attend to persistent memory
        - Input chunk can attend to memory values
        
        Args:
            persistent_size: Size of persistent memory
            memory_size: Size of memory values
            chunk_size: Size of input chunk
            
        Returns:
            Attention mask tensor
        """
        total_size = persistent_size + memory_size + chunk_size
        
        # Start with fully masked tensor (no attention allowed)
        mask = torch.ones(total_size, total_size, dtype=torch.bool)
        
        # Allow attention within persistent memory
        mask[:persistent_size, :persistent_size] = False
        
        # Allow attention within memory values
        start_idx = persistent_size
        end_idx = persistent_size + memory_size
        mask[start_idx:end_idx, start_idx:end_idx] = False
        
        # Allow attention within input chunk
        start_idx = persistent_size + memory_size
        end_idx = total_size
        mask[start_idx:end_idx, start_idx:end_idx] = False
        
        # Allow memory values to attend to persistent memory
        start_idx = persistent_size
        end_idx = persistent_size + memory_size
        mask[start_idx:end_idx, :persistent_size] = False
        
        # Allow input chunk to attend to persistent memory and memory values
        start_idx = persistent_size + memory_size
        end_idx = total_size
        mask[start_idx:end_idx, :end_idx] = False
        
        return mask 