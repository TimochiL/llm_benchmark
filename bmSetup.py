import torch
from huggingface_hub import interpreter_login

torch.cuda.empty_cache()

print(f"CUDA-enabled GPU?: {torch.cuda.is_available()}")

interpreter_login()

