# Use Hugging Face's Tokenizer
from transformers import AutoTokenizer
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Read saved text corpus
with open("human_chat.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# Tokenize data
tokenized_texts = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")

# Save tokenizer
tokenizer.save_pretrained("./tokenized_data")

# Save tokenized data
torch.save(tokenized_texts, "./tokenized_data/tokenized_texts.pt")