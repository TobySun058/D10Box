import os
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

# Folder containing images
image_folder = '.\\drawings\\Creston_Dement_Public_Library'

# Prepare a list to hold the images
all_pixel_values = []

# Iterate through all the images in the folder and load them
for image_filename in os.listdir(image_folder):
    if image_filename.endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
        image_path = os.path.join(image_folder, image_filename)
        
        # Load and preprocess the image
        pixel_values = load_image(image_path).to(device)  # Ensure it's on the correct device
        all_pixel_values.append(pixel_values)

# Define function to process images in batches of 30
def process_images_in_batches(images, batch_size=30):
    # Process in batches of `batch_size`
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        pixel_values_batch = torch.cat(batch, dim=0)  # Concatenate images into a single batch

        # Ask the model to identify the signage and page number for the batch of images
        question = '<image>\nIdentify the signage and tell me the page number at the bottom right corner.'
        
        # Define generation config
        generation_config = {'max_new_tokens': 1024, 'do_sample': True}

        # Get response from the model
        response = model.chat(tokenizer, pixel_values_batch, question, generation_config)

        # Print the response for this batch
        print(f'Response for batch starting from image {i + 1}: {response}')

# Process all the images in batches of 30
process_images_in_batches(all_pixel_values, batch_size=10)
