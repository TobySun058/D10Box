import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from ada_dataset import load_ada_sign_dataset

# === Config ===
model_path = "./models/InternVL3-8B"
output_dir = "./models/internvl-finetuned-ada-manual"
batch_size = 2
num_epochs = 10
learning_rate = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model + Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if model.generation_config.pad_token_id is None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id

# === Inject LoRA ===
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# === Preprocess ===
dataset = load_ada_sign_dataset()

def tokenize(example):
    prompt = f"<image>\n{example['instruction']}\n{example['response']}"
    enc = tokenizer(prompt, truncation=True, max_length=512, return_tensors="pt")
    return {"input_ids": enc.input_ids.squeeze(0), "attention_mask": enc.attention_mask.squeeze(0)}

tokenized = [tokenize(ex) for ex in dataset]
input_ids_list = [ex["input_ids"] for ex in tokenized]
attention_mask_list = [ex["attention_mask"] for ex in tokenized]

# === DataLoader ===
def collate(batch):
    input_ids = pad_sequence([x for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = input_ids != tokenizer.pad_token_id
    labels = input_ids.clone()
    return input_ids, attention_mask, labels

data = list(zip(input_ids_list, attention_mask_list))
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate([i[0] for i in x]))

# === Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# === Train
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for step, batch in enumerate(dataloader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# === Save LoRA model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Saved fine-tuned model to {output_dir}")
