import pandas as pd
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
import torch



data = pd.read_csv('Full Data\handm.csv')
data = data.drop(columns=['materials', 'brandName', 'url', 'stockState', 'comingSoon', 'isOnline', 'colors', 'colorShades', 'newArrival', 'mainCatCode'])

docs = []
for i in range(len(data)):
    t = f"{data.iloc[i]['productName']} - {data.iloc[i]['price']} - {data.iloc[i]['colorName']} - {data.iloc[i]['productId']} - {data.iloc[i]['details']}"
    docs.append(t)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": device}  # Move the model to GPU
)

# Storing the data in the Qdrant VectorDB
qdrant = Qdrant.from_texts(
    docs,
    embeddings,
    path="/rag_articles",
    collection_name="articles",
)


# Load DistilGPT-2 Model and Tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2", device_map="auto")

tokenizer.pad_token = tokenizer.eos_token
model.to(device)

data = load_dataset("neuralwork/fashion-style-instruct", split="train")

inputs = []
instruction = "Below given is the user's requirements. Give fashion recommendation based on this. These are the requirements - "
for text, completion in zip(data['input'], data['completion']):
    t = f"{instruction}{text} \\n {completion}"
    inputs.append(t)

data = data.add_column("text_column", inputs)
data = data.remove_columns(["input", "completion", "context"])

def tokenize_dataset(ds):
    result = tokenizer(ds["text_column"], truncation=True, max_length=512, padding="max_length")
    return result

tokenized_data = data.map(tokenize_dataset, batched=True)

lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    target_modules=["c_attn", "c_proj"],  # Correct target modules for DistilGPT-2
    task_type="CAUSAL_LM",  # Task type
)

training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,  # Enable mixed precision training (requires GPU)
    save_total_limit=2,
    push_to_hub=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_data,  # Pass the tokenized dataset directly
    args=training_args,
    peft_config=lora_config,
)

# Start fine-tuning (optional)
trainer.train()

input_text = "What should I wear for a summer party?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

model.save_pretrained("./distilgpt2-finetuned")
tokenizer.save_pretrained("./distilgpt2-finetuned")

# Load the model and tokenizer
# model = AutoModelForCausalLM.from_pretrained("./distilgpt2-finetuned", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("./distilgpt2-finetuned")