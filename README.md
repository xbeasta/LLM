<h1>Custom LLM</h1>

<h2>Step 1: Set Up Your Environment</h2>
<p>brew install python/p>
<p>pip install torch transformers datasets tokenizers tqdm</p>

<h2>Step 2: Project Structure</h2>
<ul>
    <li>download_data.py</li>
    <li>tokenize_data.py</li>
    <li>train_model.py</li>
    <li>generate_text.py</li>
</ul>

<h2>Step 3: Download Training Data </h2>

from datasets import load_dataset

# Load Wikipedia dataset
dataset = load_dataset("wikipedia", "20220301.en", split="train")

# Save text data
with open("wikipedia_corpus.txt", "w", encoding="utf-8") as f:
    for article in dataset["text"]:
        f.write(article + "\n")
