import os
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re
from copy import deepcopy

# === Preprocessing ===
def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(torch.bfloat16)

def extract_json_from_response(response):
    # Remove code block formatting (```json ... ```)
    response = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()

    # If it's a plain quoted string like "no ADA sign in this page"
    if re.fullmatch(r'"[^"\n]*"', response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return response.strip()

    # Attempt to fix common malformed JSON
    response = re.sub(r'"\s*([\w_]+)"\s*:', r', "\1":', response)  # Insert missing commas
    response = re.sub(r'^{,\s*', '{', response)  # Leading comma
    response = re.sub(r',\s*}', '}', response)   # Trailing comma

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"[!] JSON decode error: {e}")
        print(f"[!] Failed response:\n{response}\n")
        return None

def load_model(model_name, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={'': device}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model.eval(), tokenizer

# === Runtime config ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model, tokenizer = load_model(r'./models/InternVL3-8B', device=device)

# === Few-shot ADA examples ===
example_dir = "./drawings/ADA_Examples"
example_descriptions = [
    "This is a room number sign labeled 'C233' with raised text and Braille.",
    "This is a restroom sign labeled 'MEN' with a wheelchair icon and Braille.",
    "This is an occupancy load sign showing 'OCCUPANCY LOAD 49' with Braille.",
    "This is a 6x6 room name sign with raised text and Braille.",
    "This is a brushed metal office sign labeled 'AIRPORT OFFICE 101' with Braille.",
    "This is a restroom sign with male/female symbols and tactile letters.",
    "This is a square room sign labeled 'ROOM 212' with Braille at the bottom.",
    "This is a sign for building access, labeled 'ACCESSIBLE ENTRANCE' with icon and Braille.",
    "This is a public notice sign with tactile lettering and Braille.",
    "This is a safety instruction sign in high contrast, with Braille and raised text."
]

print("Seeding history with 10 ADA examples...")
seed_history = None
generation_config = {
    'max_new_tokens': 512,
    'temperature': 0.2,
    'top_p': 0.9,
    'repetition_penalty': 1.2,
    'do_sample': False
}

for i in range(1, 11):
    image_path = os.path.join(example_dir, f"ADA_Signs_{i}.png")
    pixel_values = load_image(image_path).to(device)
    question = f"<image>\n{example_descriptions[i-1]}"
    with torch.no_grad():
        response, seed_history = model.chat(
            tokenizer, pixel_values, question, generation_config,
            history=seed_history, return_history=True
        )
    print(f"Seeded example {i}/10.")

print("✅ Finished seeding history.\n")

# === Output ===
image_folder = r'./drawings/Creston_Dement_Public_Library'
txt_file = "ada_signage_results.txt"

with open(txt_file, mode='w', encoding='utf-8') as f_out:
    for image_filename in sorted(os.listdir(image_folder)):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nProcessing: {image_filename}")
            image_path = os.path.join(image_folder, image_filename)
            pixel_values = load_image(image_path).to(device)

            question = """<image>
You're analyzing an architectural drawing.

Only return ADA-compliant signage if clearly visible in the image.

If any signage is present, return:
{
  "type": "restroom | room_number | occupancy_load | other",
  "text_or_symbols": "...",
  "position_in_image": "top-left | center | bottom-right | ...",
  "sheet_number": "..."
}

If no ADA-compliant signage is visible, respond with:
"no ADA sign in this page"

Only return a valid JSON object or a quoted string, and nothing else.
"""

            with torch.no_grad():
                response, _ = model.chat(
                    tokenizer, pixel_values, question,
                    generation_config, history=deepcopy(seed_history), return_history=True
                )

            print(f"[Raw response for {image_filename}]:\n{response}\n")
            f_out.write(f"Image: {image_filename}\n")
            f_out.write(f"[Raw model response]\n{response}\n")

            parsed = extract_json_from_response(response)
            if isinstance(parsed, dict):
                page_number = parsed.get("sheet_number", "")
                ada_signs = [parsed] if "type" in parsed else "no ADA sign in this page"
            elif isinstance(parsed, str) and "no ADA sign" in parsed.lower():
                page_number = ""
                ada_signs = "no ADA sign in this page"
            else:
                page_number = ""
                ada_signs = "could not parse"

            f_out.write(f"Page Number: {page_number}\n")
            f_out.write(f"ADA Signs: {json.dumps(ada_signs, ensure_ascii=False)}\n")
            f_out.write("-" * 60 + "\n")

            print(f"✅ Saved results for {image_filename} to TXT.")
