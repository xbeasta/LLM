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

Step 2: Data Collection
To train an LLM, you'll need a large text corpus.

2.1 Download Open Datasets
You can use datasets from:
The Pile (EleutherAI)
Common Crawl (OSCAR)
Wikipedia

from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train")

2.2 Preprocess Data
Convert data to a clean text format

import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = re.sub(r'\[[0-9]*\]', '', text)  # Remove citations
    return text.strip()

dataset = [clean_text(d) for d in dataset['text']]
Step 3: Tokenization

Tokenization converts text into numerical representations.


3.1 Train a Custom Tokenizer
Use the tokenizers library to train a Byte-Pair Encoding (BPE) tokenizer:

from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["your_corpus.txt"], vocab_size=50_000, min_frequency=2)
tokenizer.save_model("tokenizer")


3.2 Load and Use the Tokenizer
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer/tokenizer.json")
tokens = tokenizer.encode("Hello world")
print(tokens)


Step 4: Define the LLM Architecture

Use PyTorch and Hugging Face’s transformers.

4.1 Create a Transformer Model
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(query, key, value)[0]
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

  
4.2 Create a Full Model


class LLM(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(LLM, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out)

        return self.fc_out(out)

        
Step 5: Train the Model

5.1 Prepare Training Data
Convert tokenized text to tensors:

import torch

def encode_texts(texts, tokenizer, seq_length):
    input_ids = [tokenizer.encode(text, max_length=seq_length, truncation=True) for text in texts]
    return torch.tensor(input_ids)

train_data = encode_texts(dataset, tokenizer, seq_length=512)
5.2 Define Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LLM(vocab_size=50_000, embed_size=512, num_layers=6, heads=8, 
            device=device, forward_expansion=4, dropout=0.1, max_length=512).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

def train(model, data, epochs):
    model.train()
    for epoch in range(epochs):
        for batch in data:
            batch = batch.to(device)
            output = model(batch)
            loss = loss_fn(output.view(-1, 50_000), batch.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}: Loss {loss.item()}")

train(model, train_data, epochs=3)
Step 6: Fine-Tuning

Fine-tune the model on specific tasks, such as chatbots or summarization.

Example fine-tuning for text generation:

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data
)

trainer.train()
Step 7: Deploy and Use the Model

Save and load the trained model:

torch.save(model.state_dict(), "llm.pth")

model.load_state_dict(torch.load("llm.pth"))
model.eval()
Generate text:

input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
output = model(input_ids)
decoded_text = tokenizer.decode(torch.argmax(output, dim=-1)[0])
print(decoded_text)

