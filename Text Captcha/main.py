from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import torch
from PIL import Image

# Load model and processor
processor = TrOCRProcessor.from_pretrained("anuashok/ocr-captcha-v3")
model = VisionEncoderDecoderModel.from_pretrained("anuashok/ocr-captcha-v3")

i=1
while i in range(13):
	# Define the image path
	image_path = f'photo ({i}).jpg'  # Ensure this path is correct

	# Load image with transparency (RGBA)
	image = Image.open(image_path).convert("RGBA")

	# Create white background and combine
	background = Image.new("RGBA", image.size, (255, 255, 255))
	combined = Image.alpha_composite(background, image).convert("RGB")

	# Prepare image for the model
	pixel_values = processor(combined, return_tensors="pt").pixel_values

	# Generate text
	generated_ids = model.generate(pixel_values)
	generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
	print(f"photo ({i}) with the text: {generated_text}")
	i=i+1