import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# Function to load and preprocess the image
def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    # Resize and normalize the image (simplified preprocessing)
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    # Add batch dimension and convert to bfloat16
    return transform(image).unsqueeze(0).to(torch.bfloat16)

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

# Set device for GPU/CPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model and tokenizer
model, tokenizer = load_model('.\models\InternVL3-8B', device=device)

# Load and preprocess image
image_path = 'image1.jpg'  # Path to your image
pixel_values = load_image(image_path).to(device)  # Make sure it's on the correct device

# Ask the model to describe the image
# question = '<image>\nWhat is in this image?'
question = 'Do you know what is ADA sign in architectural drawings? Give me speicific definition.'

# Define generation config
generation_config = {'max_new_tokens': 1024, 'do_sample': True}

# Get response from the model
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')
