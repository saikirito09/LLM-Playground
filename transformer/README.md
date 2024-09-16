# Attention Is All You Need

## Overview

This repository contains the code and resources for the research paper "Attention Is All You Need," which introduces the Transformer model, a novel neural network architecture for sequence transduction tasks like machine translation. The Transformer replaces the need for recurrent or convolutional neural networks with a model entirely based on attention mechanisms, achieving superior results while significantly reducing training time and computational requirements.

## Key Contributions

### 1. The Transformer Architecture

The Transformer model is built on a new approach that relies solely on attention mechanisms to capture global dependencies between input and output sequences. The core components of the Transformer are:

- **Encoder-Decoder Structure**: The Transformer follows the traditional encoder-decoder structure used in sequence transduction models. Both the encoder and decoder are composed of multiple identical layers stacked on top of each other.
  - **Encoder**: The encoder is a stack of 6 identical layers, each containing two main components: a multi-head self-attention mechanism and a position-wise feed-forward network.
  - **Decoder**: The decoder also consists of 6 identical layers, with an additional attention layer that focuses on the encoder's output to facilitate the generation of output sequences.

### 2. Attention Mechanisms

The Transformer introduces two key types of attention mechanisms:

- **Scaled Dot-Product Attention**: This attention mechanism calculates the relevance of different positions in the input sequence by computing the dot product of queries and keys, scaling by the dimension size, and applying a softmax function to derive attention weights.

- **Multi-Head Attention**: Instead of a single attention function, the model uses multiple attention heads that project queries, keys, and values linearly into lower-dimensional spaces, allowing the model to focus on different parts of the input sequence simultaneously. This design improves the model's ability to capture various aspects of the input.

### 3. Positional Encoding

Since the Transformer does not inherently capture the order of the input tokens (as it lacks recurrence or convolution), positional encodings are added to the input embeddings to provide information about the relative or absolute position of tokens. The paper uses sinusoidal positional encodings that allow the model to extrapolate to sequence lengths longer than those encountered during training.

### 4. Advantages of Self-Attention

The Transformer leverages self-attention mechanisms, which provide several benefits over traditional recurrent and convolutional layers:

- **Parallelization**: Self-attention allows for more parallelization than RNNs, enabling faster training, particularly for longer sequences.
- **Shorter Path Lengths**: Self-attention provides shorter paths between long-range dependencies, making it easier for the model to learn these dependencies.
- **Lower Computational Complexity**: The complexity of self-attention is lower than that of recurrent layers, especially when sequence length is less than the representation dimension.

## Results

### Machine Translation

The Transformer model has been evaluated on standard machine translation benchmarks:

- **English-to-German Translation**: Achieved a BLEU score of 28.4, outperforming all previously reported models and ensembles by more than 2 BLEU points.
- **English-to-French Translation**: Set a new state-of-the-art BLEU score of 41.8, using significantly less training time and computational resources compared to previous models.

### Generalization to Other Tasks

The paper also demonstrates the generalizability of the Transformer to other tasks beyond translation:

- **English Constituency Parsing**: The model achieved competitive results, outperforming most existing models without the need for task-specific tuning.

## Training and Implementation Details

### Data and Batching

- The model was trained on the WMT 2014 English-German dataset (4.5 million sentence pairs) and the WMT 2014 English-French dataset (36 million sentence pairs).
- Sentences were encoded using byte-pair encoding (BPE) with vocabularies of approximately 37,000 and 32,000 tokens, respectively.

### Hardware and Training Time

- Training was conducted on a single machine with 8 NVIDIA P100 GPUs.
- The base model was trained for 100,000 steps (12 hours), while the larger "big" model was trained for 300,000 steps (3.5 days).

### Optimization and Regularization

- The model uses the Adam optimizer with specific hyperparameters.
- Regularization techniques such as dropout and label smoothing were applied to improve generalization.

## Repository Contents

- **Model Code**: Implementation of the Transformer model using TensorFlow, including training and evaluation scripts.
- **Data Processing**: Scripts for data preprocessing, such as tokenization and byte-pair encoding.
- **Results**: Scripts to reproduce the results presented in the paper.

## References

For more detailed information, refer to the original paper:

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. In *31st Conference on Neural Information Processing Systems (NIPS 2017)*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
