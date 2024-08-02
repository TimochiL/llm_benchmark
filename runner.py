from llmAbs import llmAbs
import logging
import torch

"""
    Before you start:
    - py -m pip install --upgrade setuptools pip wheel
    - pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    - pip install cupy-cuda12x
    - This should be the first pip install (other packages will install CPU torch as dependency)
    - Install Visual Studio 2022 Community with c++ and python tools, then add msvc x64 to path
    
"""

def main():
    llmAbs('meta-llama/Llama-2-7b-chat-hf')

if __name__ == '__main__':
    main()
