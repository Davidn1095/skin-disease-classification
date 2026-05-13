import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "data"
INPUT_DIR = DATA_DIR / "train"
OUTPUT_DIR = DATA_DIR / "augmented_train"

AUGMENTED_PER_IMAGE = 3
IMG_SIZE = (224, 224)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_image(img):
    img = img.resize(IMG_SIZE)

    # small rotation
    angle = random.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False)

    # random horizontal flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)

    # slight brightness change
    brightness = random.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness)

    # slight contrast change
    contrast = random.uniform(0.9, 1.1)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # slight zoom/crop
    zoom = random.uniform(1.0, 1.15)
    w, h = img.size
    new_w, new_h = int(w / zoom), int(h / zoom)

    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)

    img = img.crop((left, top, left + new_w, top + new_h))
    img = img.resize(IMG_SIZE)

    return img


for class_name in os.listdir(INPUT_DIR):
    class_input_path = os.path.join(INPUT_DIR, class_name)

    if not os.path.isdir(class_input_path):
        continue

    class_output_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_path, exist_ok=True)

    print(f"Augmenting: {class_name}")

    for filename in os.listdir(class_input_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image_path = os.path.join(class_input_path, filename)

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Skipped {filename}: {e}")
            continue

        base_name = os.path.splitext(filename)[0]

        for i in range(AUGMENTED_PER_IMAGE):
            aug_img = augment_image(img)
            save_name = f"{base_name}_aug_{i+1}.jpg"
            save_path = os.path.join(class_output_path, save_name)
            aug_img.save(save_path, quality=95)

print("Done. Augmented images saved in:", OUTPUT_DIR)