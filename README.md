<h1>Custom LLM</h1>
Step 1: Set Up Your Environment

1.1 Install Python and Dependencies
brew install python3
pip3 install torch transformers datasets tokenizers accelerate sentencepiece numpy pandas tqdm

1.2 Install CUDA (if using GPU)
MacOS doesn’t support NVIDIA CUDA, but if you’re using an external Linux server with GPUs:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For Apple Silicon (M1/M2), install Metal Performance Shaders:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl
