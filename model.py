import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset

# Define a small GPT model
config = GPT2Config(
    vocab_size=30522,
    n_embd=512,
    n_layer=6,
    n_head=8
)
model = GPT2LMHeadModel(config)
model.train()

tokenized_texts = torch.load("tokenized_data/tokenized_texts.pt")

input_ids = tokenized_texts["input_ids"]
attention_masks = tokenized_texts["attention_mask"]

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_dict({"input_ids": input_ids,
                                   "attention_mask": attention_masks
                                   })

training_args = TrainingArguments(
    output_dir="./llm_checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=50,
    logging_dir="./logs"
)

# Define Data Collator for Language Modeling
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, # Set `mlm=True` if using masked LM like BERT
    pad_to_multiple_of=8  # Optimizes memory usage
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_dataset,
    data_collator=data_collator,
    tokenizer = tokenizer
)
trainer.train()

# Save model
model.save_pretrained("./my_llm")