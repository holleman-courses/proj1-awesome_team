from PIL import Image, ImageTransform
import os

def resize_image(in_path: str, out_path: str, size: tuple):
    with Image.open(in_path) as img:
        img.resize(size).save(out_path)

for root, _, files in os.walk('./dataset'):
    for file in files:
        in_path = os.path.join(root, file)
        out_path = os.path.join('./dataset/resized', root, file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        resize_image(in_path, out_path, (256, 256))
            