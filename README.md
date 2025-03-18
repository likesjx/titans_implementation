# Titans AI Architecture Implementation

This repository contains a Python implementation of Google's Titans AI architecture based on the paper "Titans: Learning to Memorize at Test Time" by Ali Behrouz et al. 

## Overview

Titans architecture introduces a novel approach to memory mechanisms in transformer-based models by implementing three levels of memory:

1. **Short-term memory**: Similar to standard transformer attention mechanisms
2. **Long-term memory**: A neural network that learns what to remember, when to remember, and how to forget (at test time)
3. **Persistent memory**: Learnable parameters that store task-specific information

The key innovation in Titans is the implementation of "online learning" or "test-time learning" where the model's long-term memory module is updated during inference using gradient descent based on a "surprise" metric.

## Project Structure

```
titans_implementation/
├── configs/         # Configuration files
├── data/            # Data handling utilities and datasets
├── models/          # Model implementations
├── scripts/         # Training and inference scripts
├── utils/           # Utility functions
├── README.md        # This readme file
└── requirements.txt # Project dependencies
```

## Memory Mechanisms

### Short-term Memory
Implemented using standard self-attention mechanisms as in traditional transformers.

### Long-term Memory
A separate neural network (MLP) that:
- Updates based on gradient descent at test time
- Uses a "surprise" metric as the loss function
- Incorporates momentum to extend memorization over time
- Includes a forgetting mechanism to prevent memory overflow

### Persistent Memory
Implemented as additional learnable parameters appended to the start of every sequence.

## Implementation Approaches

This implementation explores the different ways to integrate memory into the architecture as described in the paper:

1. **Memory as a Context (MAC)**: Memory retrieval is added to the input
2. **Memory as a Gate (MAG)**: Memory controls the output via a gating mechanism
3. **Memory as a Layer (MAL)**: Memory serves as a preprocessing step
4. **Long-term Memory Module (LMM)**: A purely recurrent approach without self-attention

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following Python packages:

#### Core Dependencies
- `torch>=2.0.0`: PyTorch for deep learning operations
- `transformers>=4.30.0`: Hugging Face Transformers library
- `numpy>=1.24.0`: Numerical computing library
- `tqdm>=4.65.0`: Progress bar library
- `wandb>=0.15.0`: Weights & Biases for experiment tracking
- `matplotlib>=3.7.0`: Plotting library
- `pandas>=2.0.0`: Data manipulation library
- `scikit-learn>=1.3.0`: Machine learning utilities
- `tensorboard>=2.13.0`: TensorBoard for visualization
- `accelerate>=0.20.0`: Accelerate for distributed training
- `bitsandbytes>=0.41.0`: Quantization utilities
- `sentencepiece>=0.1.99`: Tokenization library
- `protobuf>=4.23.0`: Protocol buffers
- `safetensors>=0.3.1`: Safe tensor serialization
- `einops>=0.6.1`: Einstein notation operations
- `datasets>=2.12.0`: Hugging Face datasets library
- `evaluate>=0.4.0`: Hugging Face evaluation library
- `rouge>=1.0.1`: ROUGE metrics for text generation
- `nltk>=3.8.1`: Natural Language Toolkit
- `sacrebleu>=2.3.1`: BLEU score calculation
- `bert-score>=0.3.13`: BERT-based evaluation metrics
- `pycocoevalcap>=1.2`: COCO caption evaluation metrics
- `spacy>=3.6.0`: Natural language processing library
- `networkx>=3.1`: Graph theory and network algorithms
- `scipy>=1.10.0`: Scientific computing library

#### Development Dependencies
- `pytest>=7.3.1`: Testing framework
- `black>=23.3.0`: Code formatter
- `isort>=5.12.0`: Import sorter
- `flake8>=6.0.0`: Code linter
- `mypy>=1.3.0`: Static type checker

#### Testing Dependencies
- `pytest-cov>=4.0.0`: Test coverage reporting
- `pytest-mock>=3.10.0`: Mocking utilities for tests
- `pytest-asyncio>=0.21.0`: Async test support
- `pytest-timeout>=2.1.0`: Test timeout utilities
- `pytest-xdist>=3.3.0`: Distributed test execution
- `pytest-benchmark>=4.0.0`: Benchmarking utilities
- `pytest-randomly>=3.13.0`: Random test ordering
- `pytest-sugar>=0.9.7`: Enhanced test output
- `pytest-html>=4.1.1`: HTML test reports
- `pytest-metadata>=3.0.0`: Test metadata collection
- `pytest-env>=1.0.1`: Environment variable management

### Usage

[Coming soon]

## References

- Behrouz, A. et al. (2024). "Titans: Learning to Memorize at Test Time"

## License

[MIT License]
