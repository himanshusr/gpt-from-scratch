# GPT from Scratch

## Team Members (Group 4)
- Vishruth Byaramudu Lokesh
- Zeba Samiya
- Himanshu Singh Rao
- Akshay Aralikatti

## Overview
This repository contains implementations of a Generative Pre-trained Transformer (GPT) model built incrementally from scratch using PyTorch. The primary objective is educational, providing a clear, step-by-step understanding of how transformer architectures are developed—from simple bigram models to advanced GPT architectures incorporating attention mechanisms, residual connections, and layer normalization.

## Files and Descriptions

- **`bigram.py`**: Implements the simplest bigram language model, predicting the next character based on the current one.

- **`bigramV2_single_head_attention.py`**: Introduces a single-head self-attention mechanism to improve contextual understanding.

- **`bigramV2_multi_head_attention.py`**: Extends to multi-head attention, enhancing the model's ability to capture diverse linguistic relationships.

- **`bigramV2_multi_head_attention_FF.py`**: Adds a feedforward neural network for deeper representation capabilities.

- **`bigramV2_multi_head_attention_FF_block_residual.py`**: Incorporates residual connections to mitigate vanishing gradients and improve training stability.

- **`bigramV2_multi_head_attention_FF_block_residual_layernorm.py`**: Adds layer normalization for further training stability and improved generalization.

- **`bigramV2_multi_head_attention_FF_block_residual_layernorm_scaling_up.py`**: Scales up the model significantly, increasing depth, width, and training efficiency.

- **`harry_potter_with_saved_weights.py`**: Trains a scaled-up GPT model specifically on the Harry Potter dataset, demonstrating style adaptation.

- **`inference.py`**: Provides a simple script for loading the trained GPT model and generating text from a given prompt.

## Dataset
- **Shakespeare Dataset**: Utilized initially for demonstrating basic transformer concepts.
- **Harry Potter Dataset**: Used to show the capability of the model to adapt to different literary styles.

## Usage

### Training

Run the desired Python script to train the respective model:

```bash
python bigram.py
```

Similarly, execute other files (`bigramV2_single_head_attention.py`, `bigramV2_multi_head_attention.py`, etc.) as required.

### Inference

After training, generate text using:

```bash
python inference.py
```

You can specify prompts directly in the script for customized text generation.

## Dependencies
- Python 3.x
- PyTorch

Install PyTorch from [here](https://pytorch.org/get-started/locally/).

```bash
pip install torch
```

## References
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
