# GPT From Scratch

A small, from-scratch implementation of a decoder-only Transformer
language model built while following Andrej Karpathy's video.

This is a character level language model trained on the Tiny Shakespeare
dataset. It learns to predict the next character in a sequence, and once
trained, generates new Shakespeare style (not coherent Shakespeare) text
one character at a time.

The goal of this project was to understand every component of a Transformer
by implementing it directly in PyTorch without relying on any pre-built
attention or transformer libraries.

## What's implemented

- Character-level tokenization
- Train/validation data split and batching
- Self-attention (single head), implemented from the underlying math up
- Multi-head attention
- Feedforward layers
- Residual connections
- Layer normalization
- Stacked Transformer blocks
- Token + positional embeddings
- A full training loop
- Autoregressive text generation

## Setup

```bash
pip install torch
```

Download the training data (Tiny Shakespeare):

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Run

```bash
python gpt.py
```

This will train the model for 5000 steps (a few minutes on CPU, faster on
GPU) and print the loss every 500 steps, followed by 500 characters of
generated text.

## Example output

```
step 0: loss 4.3826
step 500: loss 2.3123
step 1000: loss 1.9975
...
step 4500: loss 1.7483
final loss: 1.6216325759887695
```

Sample generated text after training:

```
Tell carloud! I'll Plast enly of put consucle my forst made
To give with in of Bleath an That, bude to batched; and they
thready to it...
```

Not coherent English, but it has clearly learned the shape of the text:
character names followed by colons, sentence capitalization, plausible
punctuation, and word-length chunks, none of which a simpler model (e.g. a
bigram model using only the previous character) can produce.

## Model size

The default hyperparameters (`n_embd=64`, `n_head=4`, `n_layer=4`,
`block_size=32`) produce a small model that trains quickly, intended for
learning and experimentation rather than high-quality output. Increasing
these values (Karpathy's video uses `n_embd=384`, `n_layer=6`, `n_head=6`,
`block_size=256`) produces noticeably more coherent text, but requires a GPU
to train in a reasonable amount of time.

## Acknowledgements

Architecture and training approach follow Andrej Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) and
[ng-video-lecture](https://github.com/karpathy/ng-video-lecture) repositories.
