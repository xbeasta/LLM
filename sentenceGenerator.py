from transformers import GPT2LMHeadModel, AutoTokenizer

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./my_llm")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("\n")
input_text = input("Your Input: ")

tokenized_inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(
    input_ids = tokenized_inputs.input_ids,
    attention_mask = tokenized_inputs.attention_mask,
    pad_token_id = tokenizer.eos_token_id,
    max_length=100,
    do_sample=True,
    top_k=100,
    temperature=0.4,
    repetition_penalty=1.2,
    early_stopping=True,
)

print("\n")
print(tokenizer.batch_decode(output)[0])
print("\n")
# print(output[0][0])