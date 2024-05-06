import os
import random


def select_random_image(folder_path: str) -> str | None:
    files = os.listdir(folder_path)
    images = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if images:
        random_image = random.choice(images)
        print(f"Selected random image: {random_image}")
        return os.path.join(folder_path, random_image)
    else:
        print("No images found in the directory.")
        return None
