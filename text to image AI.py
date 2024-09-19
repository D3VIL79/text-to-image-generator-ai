import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Replace 'YOUR_HUGGINGFACE_API_TOKEN' with your actual Hugging Face API token
HUGGINGFACE_TOKEN = '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'

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=HUGGINGFACE_TOKEN)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def text_to_image(text, output_path='output_image.png'):
    # Generate image
    image = pipe(text).images[0]
    
    # Save image
    image.save(output_path)
    print(f"Image saved as {output_path}")

if __name__ == "__main__":
    # Get text input from the user
    text = input("Enter the text description for the image: ")
    
    # Generate image based on user input
    text_to_image(text)
