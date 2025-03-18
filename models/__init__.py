from .base_model import TitansBaseModel
from .memory_as_context import MemoryAsContextModel
from .memory_as_gate import MemoryAsGateModel
from .memory_as_layer import MemoryAsLayerModel
from .long_term_memory_module import LongTermMemoryModule
from .long_term_memory import LongTermMemoryMLP, ParallelLongTermMemory

# Model registry for easy access to model variants
MODEL_REGISTRY = {
    "mac": MemoryAsContextModel,  # Memory as a Context
    "mag": MemoryAsGateModel,     # Memory as a Gate
    "mal": MemoryAsLayerModel,    # Memory as a Layer
    "lmm": LongTermMemoryModule   # Long-term Memory Module (pure recurrent)
}

def get_model(model_type, **kwargs):
    """
    Factory function to get a model by name.
    
    Args:
        model_type: String identifier of the model type ("mac", "mag", "mal", or "lmm")
        **kwargs: Model configuration parameters
        
    Returns:
        Instantiated model
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_type](**kwargs)

__all__ = [
    'TitansBaseModel',
    'MemoryAsContextModel',
    'MemoryAsGateModel',
    'MemoryAsLayerModel',
    'LongTermMemoryModule',
    'LongTermMemoryMLP',
    'ParallelLongTermMemory',
    'get_model',
    'MODEL_REGISTRY'
]

# Empty file to make the directory a Python package
