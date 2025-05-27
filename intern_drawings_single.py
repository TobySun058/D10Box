import os
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import json
import re

# Function to load and preprocess the image
def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(torch.bfloat16)

# Function to extract and clean JSON from model response
def extract_json_from_response(response):
    try:
        # Remove markdown code block if it exists
        match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try to parse directly
        return json.loads(response)
    except Exception:
        return None

# Setup model loading
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

    return model, tokenizer

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model, tokenizer = load_model(r'./models/InternVL3-8B', device=device)

# Image folder and output CSV
image_folder = r'./drawings/Creston_Dement_Public_Library'
csv_file = "ada_signage_results.csv"

# Prepare CSV output
with open(csv_file, mode='w', newline='', encoding='utf-8') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=["image", "page_number", "ada_signs"])
    writer.writeheader()

    # Loop over images
    for image_filename in sorted(os.listdir(image_folder)):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_filename)
            pixel_values = load_image(image_path).to(device)

            question = """<image>
            Identify any ADA-compliant signage present in the image. For each sign, return:
            {
            "type": "restroom | exit | accessible entrance | other",
            "text_or_symbols": "...",
            "position_in_image": "top-left | bottom-right | center | ..."
            }

            If no ADA-compliant signage is found, respond with:
            "no ADA sign in this page"

            Also, extract the page number (if present) from the bottom right corner and include it as:
            "page_number": "..."

            Return plain JSON only without Markdown formatting.
            """

            generation_config = {'max_new_tokens': 1024, 'do_sample': True}
            response = model.chat(tokenizer, pixel_values, question, generation_config)

            print(f'Image: {image_filename}\nResponse: {response}\n')

            parsed = extract_json_from_response(response)
            if parsed:
                page_number = parsed.get("page_number", "")
                ada_signs = [parsed] if "type" in parsed else "no ADA sign in this page"
            else:
                page_number = ""
                if "no ADA sign in this page" in response.lower():
                    ada_signs = "no ADA sign in this page"
                else:
                    ada_signs = "could not parse"

            writer.writerow({
                "image": image_filename,
                "page_number": page_number,
                "ada_signs": json.dumps(ada_signs, ensure_ascii=False)
            })
   