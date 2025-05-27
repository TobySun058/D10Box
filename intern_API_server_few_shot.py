from flask import Flask, request, jsonify
import os, torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

# === Load Image for Vision-Language Model ===
def load_image(image_file, input_size=448, device='cuda'):
    image = Image.open(image_file).convert('RGB')
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
    ])
    return transform(image).unsqueeze(0).to(device, dtype=torch.bfloat16)

# === Model Loader ===
def load_model(model_path, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model.eval(), tokenizer

# === Seeding History ===
def seed_few_shot_history(model, tokenizer, device):
    examples_dir = "./drawings/ADA_Examples"
    descriptions = [
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
    history = None
    generation_config = {
        "max_new_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "do_sample": False
    }

    for i, desc in enumerate(descriptions, 1):
        image_path = os.path.join(examples_dir, f"ADA_Signs_{i}.png")
        image = load_image(image_path, device=device)
        question = f"<image>\n{desc}"
        with torch.no_grad():
            _, history = model.chat(tokenizer, image, question, generation_config, history=history, return_history=True)

    return history

# === Initialize ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, tokenizer = load_model("./models/InternVL3-8B", device)
seed_history = seed_few_shot_history(model, tokenizer, device)

# === Flask App ===
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_tensor = load_image(image_file, device=device)

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

    generation_config = {
        'max_new_tokens': 512,
        'temperature': 0.2,
        'top_p': 0.9,
        'repetition_penalty': 1.2,
        'do_sample': False
    }

    with torch.no_grad():
        response, _ = model.chat(
            tokenizer,
            image_tensor,
            question,
            generation_config=generation_config,
            history=deepcopy(seed_history),
            return_history=True
        )

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
