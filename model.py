import torch
import math
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer, EarlyStoppingCallback
from datasets import Dataset

#
# Step 1: Define a GPT2 model
#
config = GPT2Config(
    vocab_size=50257,
    n_embd=512,
    n_layer=6,
    n_head=8
)
model = GPT2LMHeadModel(config)
model.lm_head.weight.data.normal_(mean=0.0, std=0.02)
model.train()

#
# Load tokenized data
#
tokenized_texts = torch.load("tokenized_data/tokenized_texts.pt")

#
# Convert tokenized texts to Hugging Face's Dataset
#
input_ids = tokenized_texts["input_ids"]
attention_masks = tokenized_texts["attention_mask"]
full_dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_masks})

#
# Split dataset into train and eval (80% train, 20% eval)
#s
split_dataset = full_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

#
# Step 2: Define Training Arguments
#
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save checkpoints
    evaluation_strategy="steps", # Evaluate method
    eval_steps=500, # Evaluate every 500 steps
    save_steps=500, # Save evaluation every 500 steps
    logging_dir='./logs',  # Directory for logs
    logging_steps=500,  # Log progress every 500 steps
    save_total_limit=2,  # Keep only the last 2 checkpoint
    load_best_model_at_end=True, # Only use the best performing model at the end
    learning_rate=5e-5,  # Experiment with different learning rates
    per_device_train_batch_size=8,  # Adjust based on your GPU capacity
    num_train_epochs=10,  # Start small and increase if necessary
)

#
# Define Data Collator for Language Modeling. This helps with efficiency for CPU and GPU.
#
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # Padding is on the left because GPT2 runs better this way.

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False,
    pad_to_multiple_of = 8, # Optimizes memory usage
    return_tensors = "pt"
)

#
# Step 3: Initialize Trainer
#
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    data_collator = data_collator,
    tokenizer = tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

#
# Step 4: Train the model
#
trainer.train()

#
# Function to calculate perplexity.
# Perplexity is a good method to measure language generative tasks.
# Lower values of perplexity, the better the model.
#
def calculate_perplexity(log_loss):
    return math.exp(log_loss)

#
# Evaluate the model
#
eval_results = trainer.evaluate()

#
# Calculate and display perplexity
#
perplexity = calculate_perplexity(eval_results['eval_loss'])
print(f"Perplexity: {perplexity}")

#
# Save model
#
model.save_pretrained("./my_llm")

print("Training Complete")