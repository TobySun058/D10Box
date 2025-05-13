from transformers import pipeline, LlavaForConditionalGeneration, AutoProcessor, AutoImageProcessor, AutoTokenizer

# Load the model with ignore_mismatched_sizes=True
model = LlavaForConditionalGeneration.from_pretrained(
    "./models/llava-1.5-7b-hf",
    ignore_mismatched_sizes=True
)

# Load the image processor (ensure it’s the correct one for the task)
image_processor = AutoImageProcessor.from_pretrained("./models/llava-1.5-7b-hf")

# Load the tokenizer manually, which includes the pad_token_id
tokenizer = AutoTokenizer.from_pretrained("./models/llava-1.5-7b-hf")

# Set pad_token_id manually if it’s missing
tokenizer.pad_token_id = tokenizer.eos_token_id  # Use eos_token_id for padding

# Use the model, image processor, and tokenizer in the pipeline
pipe = pipeline("image-text-to-text", model=model, image_processor=image_processor, tokenizer=tokenizer)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg"},
            {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},
        ],
    },
]

# Process the input and generate the output
out = pipe(text=messages, max_new_tokens=20)

# Print the output
print(out)
