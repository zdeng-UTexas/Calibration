import pyheif
from PIL import Image
import os

def convert_heic_to_jpeg(image_path, output_path):
    heif_file = pyheif.read(image_path)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
    image.save(output_path, "JPEG")

image_dir = "images"
output_dir = "converted_images"
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(image_dir):
    if file_name.endswith(".HEIC"):
        heic_path = os.path.join(image_dir, file_name)
        jpeg_path = os.path.join(output_dir, file_name.replace(".HEIC", ".jpg"))
        convert_heic_to_jpeg(heic_path, jpeg_path)
