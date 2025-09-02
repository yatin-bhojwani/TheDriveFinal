# import huggingface_hub
# import requests
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# # conditional image captioning
# text = "a photography of"
# inputs = processor(raw_image, text, return_tensors="pt")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

# # unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt")

# out = model.generate(**inputs)
# print(processor.decode(out[0], skip_special_tokens=True))

import os
import sys
from dotenv import load_dotenv
import json
from PIL import Image
import google.generativeai as genai

load_dotenv()

def generate_caption_json(image_path, api_key):
    if not os.path.exists(image_path):
        return json.dumps({"error": "Image file not found."})
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        img = Image.open(image_path)
        caption = model.generate_content(["Write a caption for the image in english", img])
        return json.dumps({"response": caption.text})
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Usage: python image_caption_json.py <image_path> <api_key>
    # if len(sys.argv) != 3:
    #     print(json.dumps({"error": "Usage: python image_caption_json.py <image_path> <api_key>"}))
    #     sys.exit(1)
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    models = genai.list_models()
    print([m.name for m in models])
    image_path = r"C:\Users\DELL\drive_PClub\implement\api\app\GraphRAG\src\__results___23_2.png"
    
    print(generate_caption_json(image_path, api_key))


