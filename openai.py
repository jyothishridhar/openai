import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
from transformers import set_seed

# Set your Hugging Face token
os.environ["HUGGINGFACE_TOKEN"] = "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"

# Log in to the Hugging Face Hub
login(token=os.environ["HUGGINGFACE_TOKEN"])

set_seed(42)

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# Define the prompts
prompts = [
    "Generate test cases for login functionality for the below conditions:",
    "Don't use email ID.",
    "Username should be unique.",
    "User password should have at least two numbers and three alphabets."
]

# Generate and print test cases in batches
batch_size = 2  # Adjust the batch size as needed
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i + batch_size]
    generator = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta")
    batch_results = generator(batch_prompts, max_length=50, num_return_sequences=2, num_beams=4, truncation=True)
    
    for prompt, results in zip(batch_prompts, batch_results):
        print(f"Prompt: {prompt}")
        for j, output in enumerate(results):
            print(f"Test case {j + 1}:\n{output['generated_text']}\n")

print("Text generation completed.")
