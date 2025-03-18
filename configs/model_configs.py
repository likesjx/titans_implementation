"""
Model configuration classes for Titans architecture variants.
"""

class ModelConfig:
    """Base class for model configurations."""
    
    def __init__(
        self,
        model_type="mac",
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072,
        max_seq_len=1024,
        persistent_memory_size=16,
        dropout=0.1,
        # Memory as Context specific
        chunk_size=64,
        # Memory as Gate specific
        window_size=128,
        # Long-term memory specific
        memory_hidden_dims=[256, 128],
        memory_learning_rate=0.01,
        momentum_factor=0.9,
        forget_factor=0.1,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.persistent_memory_size = persistent_memory_size
        self.dropout = dropout
        
        # Memory as Context specific
        self.chunk_size = chunk_size
        
        # Memory as Gate specific
        self.window_size = window_size
        
        # Long-term memory specific
        self.memory_hidden_dims = memory_hidden_dims
        self.memory_learning_rate = memory_learning_rate
        self.momentum_factor = momentum_factor
        self.forget_factor = forget_factor
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class TitansTinyConfig(ModelConfig):
    """Tiny model configuration (comparable to GPT-2 Small but smaller)."""
    
    def __init__(self, model_type="mac"):
        super().__init__(
            model_type=model_type,
            vocab_size=50257,
            hidden_size=384,
            num_layers=6,
            num_heads=6,
            ff_dim=1536,
            max_seq_len=1024
        )


class TitansSmallConfig(ModelConfig):
    """Small model configuration (comparable to GPT-2 Medium)."""
    
    def __init__(self, model_type="mac"):
        super().__init__(
            model_type=model_type,
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            ff_dim=3072,
            max_seq_len=1024
        )


class TitansMediumConfig(ModelConfig):
    """Medium model configuration (comparable to GPT-2 Large)."""
    
    def __init__(self, model_type="mac"):
        super().__init__(
            model_type=model_type,
            vocab_size=50257,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
            ff_dim=4096,
            max_seq_len=1024
        )


class TitansLargeConfig(ModelConfig):
    """Large model configuration (comparable to GPT-2 XL)."""
    
    def __init__(self, model_type="mac"):
        super().__init__(
            model_type=model_type,
            vocab_size=50257,
            hidden_size=1280,
            num_layers=36,
            num_heads=20,
            ff_dim=5120,
            max_seq_len=1024
        )


# Dictionary mapping configuration names to configuration classes
CONFIG_REGISTRY = {
    "tiny": TitansTinyConfig,
    "small": TitansSmallConfig,
    "medium": TitansMediumConfig,
    "large": TitansLargeConfig
}


def get_config(model_size="small", model_type="mac"):
    """
    Get model configuration by size and type.
    
    Args:
        model_size: One of "tiny", "small", "medium", or "large"
        model_type: One of "mac", "mag", "mal", or "lmm"
        
    Returns:
        ModelConfig instance
    """
    if model_size not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model size: {model_size}. Available sizes: {list(CONFIG_REGISTRY.keys())}"
        )
    
    return CONFIG_REGISTRY[model_size](model_type=model_type) 