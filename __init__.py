"""
Titans AI Architecture Implementation

This package implements Google's Titans architecture with three levels of memory:
1. Short-term memory: Similar to standard transformer attention mechanisms
2. Long-term memory: A neural network that learns to memorize at test time
3. Persistent memory: Learnable parameters that store task-specific information
"""

from .models import (
    TitansBaseModel,
    MemoryAsContextModel,
    MemoryAsGateModel,
    MemoryAsLayerModel,
    LongTermMemoryModule,
    get_model,
    MODEL_REGISTRY
)

from .configs.model_configs import (
    ModelConfig,
    TitansTinyConfig,
    TitansSmallConfig,
    TitansMediumConfig,
    TitansLargeConfig,
    get_config,
    CONFIG_REGISTRY
)

from .utils.tokenizer import TokenizerWrapper

__version__ = "0.1.0"
__all__ = [
    'TitansBaseModel',
    'MemoryAsContextModel',
    'MemoryAsGateModel',
    'MemoryAsLayerModel',
    'LongTermMemoryModule',
    'get_model',
    'MODEL_REGISTRY',
    'ModelConfig',
    'TitansTinyConfig',
    'TitansSmallConfig',
    'TitansMediumConfig',
    'TitansLargeConfig',
    'get_config',
    'CONFIG_REGISTRY',
    'TokenizerWrapper'
]
