<h1>Custom LLM</h1>
<p>Side Note: The code below may be outdated, refer to code in the repo for up-to-date progress.</p>

<h2>Step 1: Set Up Your Environment</h2>
<p>brew install python</p>
<p>pip install torch transformers datasets tokenizers tqdm</p>

<h2>Step 2: Project Structure</h2>
<ul>
    <li>download_data.py</li>
    <li>tokenize_data.py</li>
    <li>train_model.py</li>
    <li>generate_text.py</li>
</ul>

<h2>Step 3: Download Training Data </h2>

```
# download_data.py

from datasets import load_dataset
# Load Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train")

# Save text data
with open("wikipedia_corpus.txt", "w", encoding="utf-8") as f:
    for article in dataset["text"]:
        f.write(article + "\n")
```

<h2>Step 4: Tokenize Data</h2>

```
# tokenize_data.py

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Read corpus
with open("wikipedia_corpus.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# Tokenize data
tokenized_texts = tokenizer(texts[:1000], truncation=True, padding=True, return_tensors="pt")

# Save tokenize
tokenizer.save_pretrained("./tokenized_data")

# Save tokenized data
torch.save(tokenized_texts, "./tokenized_data/tokenized_texts.pt")
```

<h2> Step 5: Define & Train LLM</h2>

```
# train_model.py

import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

# Define a small GPT model
config = GPT2Config(
    vocab_size=30522,
    n_embd=256,
    n_layer=4,
    n_head=4
)
model = GPT2LMHeadModel(config)
model.train()

tokenized_texts = torch.load("tokenized_data/tokenized_texts.pt")

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"input_ids": tokenized_texts["input_ids"],
                                   "attention_mask": tokenized_texts["attention_mask"]})

training_args = TrainingArguments(
    output_dir="./llm_checkpoints",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs"
)
trainer = Trainer(
    model=model,
    args=training_args
)
trainer.train()

# Save model
model.save_pretrained("./my_llm")
```

<h2>Step 6: Generate Text</h2>

```
# generate_text.py

from transformers import GPT2LMHeadModel, AutoTokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./my_llm")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=50)

print(tokenizer.decode(output[0]))
```

<h2>Step 7: Connecting to AWS</h2>
<p>chmod 400 "badminton_llm_key.pem"</p>
<p>ssh -i "badminton_llm_key.pem" ubuntu@ec2-52-14-208-196.us-east-2.compute.amazonaws.com</p>
<p>scp -i /Users/elmer/Desktop/CustomProjects/badminton_llm_key.pem /Users/elmer/Desktop/CustomProjects/LLM ubuntu@ec2-52-14-208-196.us-east-2.compute.amazonaws.com</p>
<p>scp -i /Users/elmer/Desktop/CustomProjects/badminton_llm_key.pem -r /Users/elmer/Desktop/CustomProjects/LLM ubuntu@ec2-52-14-208-196.us-east-2.compute.amazonaws.com
</p>
