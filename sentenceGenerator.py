from transformers import GPT2LMHeadModel, AutoTokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./my_llm")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

input_text = "what is your plan?"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=24)

print(tokenizer.decode(output[0]))