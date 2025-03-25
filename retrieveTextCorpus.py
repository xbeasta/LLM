from datasets import load_dataset

# Load Wikipedia dataset from Hugging Face
dataset = load_dataset("wikipedia", "20220301.en", split="train")

# Save text data
with open("wikipedia_corpus.txt", "w", encoding="utf-8") as f:
    for article in dataset["text"]:
        f.write(article + "\n")