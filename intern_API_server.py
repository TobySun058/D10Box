from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup model loading
def load_model(model_name, device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 as specified
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={'': device}  # Ensure it's using the correct device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer

# Function to load and preprocess the image
def load_image(image_file, input_size=448, device='cuda'):
    image = Image.open(image_file).convert('RGB')
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # Add batch dimension and ensure the tensor is moved to the same device as the model
    return transform(image).unsqueeze(0).to(device, dtype=torch.bfloat16)

# Set device for GPU/CPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and tokenizer
model, tokenizer = load_model('.\models\InternVL3-8B', device=device)

# Flask app initialization
app = Flask(__name__)

# Route to handle image input and get model response
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Get the image from the request
    image_file = request.files['image']
    image = load_image(image_file)

    # Define the question to ask the model
    question = '<image>\nWhat is in this image?'

    # Define generation config
    generation_config = {'max_new_tokens': 1024, 'do_sample': True}

    # Get response from the model
    response = model.chat(tokenizer, image, question, generation_config)

    # Return the model's response as a JSON
    return jsonify({'response': response})

# Run the app
if __name__ == '__main__':
    app.run(debug=False)
