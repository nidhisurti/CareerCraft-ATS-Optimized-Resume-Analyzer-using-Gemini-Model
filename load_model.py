from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with your desired model name if necessary
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Example model

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example input text
input_text = "What is the capital of France?"

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate model output
outputs = model.generate(inputs['input_ids'], max_length=50)

# Decode the output
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated Text: {output_text}")
