# LLM Safety Benchmark for KV Cache Quantization

> [!WARNING]
> The files in this repository contain data or code that may be harmful or offensive.

### See the corresponding research paper here: [Key-Value Cache Quantization in Large Language Models: A Safety Benchmark](https://wepub.org/index.php/IJCSIT/article/view/4079)

## Status

> [!NOTE]
> Stable.

## Features

This benchmark evaluates the effect of KV cache quantization on LLM response safety using sample questions distributed among 13 forbidden scenarios.

> [!NOTE]
> Currently, only the Meta Llama-2 7B Chat model is implemented in the benchmark with HQQ backend.
> This benchmark serves as a proof-of-concept. Other models, model families, and backends are considerations for future work.

## Resources

- Code is adapted from [QuantizedKVCache_Generation_Transformers.ipynb](https://colab.research.google.com/drive/1YKAdOLoBPIore77xR5Xy0XLN8Etcjhui?usp=sharing)
- Benchmark is largely inspired by ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)
