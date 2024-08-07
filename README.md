# LLM Safety Benchmark for KV Cache Quantization

> [!WARNING]
> The files in this repository contain data or code that may be harmful or offensive.

### See the corresponding research paper here: [coming soon...]

## Status

> [!WARNING]
> This benchmark is still in development. Broken features are expected.

## Features

This benchmark evaluates the effect of KV Cache Quantization on safety measures against offensive and jailbreak prompts.

> [!NOTE]
> Currently, only the Meta Llama-2 7B Chat model is implemented in the benchmark with HQQ backend.
> Implementation of other models, model families, and backends are coming soon.

## Resources

- Code is adapted from [QuantizedKVCache_Generation_Transformers.ipynb](https://colab.research.google.com/drive/1YKAdOLoBPIore77xR5Xy0XLN8Etcjhui?usp=sharing)
- Benchmark is largely inspired by ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)
