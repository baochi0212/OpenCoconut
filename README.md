[![arXiv](https://img.shields.io/badge/arXiv-2412.06769-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2412.06769)

# OpenCoconut

OpenCoconut intends to replicate the Chain of Continuous Thought (COCONUT) paper that implements a novel latent reasoning paradigm. The main idea is to generate thoughts in latent space by utilizing the hidden states during prefilling before we start decoding response. We build on the public dataset from the paper for math [casperhansen/gsm8k_synthetic_cot](https://huggingface.co/datasets/casperhansen/gsm8k_synthetic_cot).

## Getting started

Install the package and then go look in `examples` for how to run training and inference.

```
git clone https://github.com/casper-hansen/OpenCoconut.git
cd OpenCoconut
pip install -e .
```

If you want to see the thoughts during training or inference, you can run with `DEBUG=1 python ...`.

## Future Work

1. Improve the loss function
    - Use a REINFORCE loss for thought tokens.
2. Implement COCONUT for pretraining
    - Scaling through pretraining would be ideal due to data availability.
3. Implement early exit with a classifier
    - Potentially as simple as training `nn.Linear(X, 1)`.
4. Improve the datasets
    - Find a good mix of step-by-step datasets for math, coding, and general domain.
5. Adaptively switch between latent and language space during decoding.
    - This could help improve accuracy by allowing generation of more thoughts.
6. Unit testing different parts of the code.
    - This should help with keeping bugs in check as code changes.
