import os
import json
from PIL import Image

def get_image_info(directory):
    image_info = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                file_path = os.path.join(root, file)
                # replace the string 'sketch_data' with 'crawler_data' in the path
                original_path = file_path.replace('sketch_data', 'crawler_data')
                with Image.open(file_path) as img:
                    resolution = img.size
                image_info.append({
                    'image': original_path,
                    'sketch': file_path,
                    'resolution': resolution
                })
    return image_info

def save_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    # get the last part of the directory path
    last = directory.split('/')[-1]
    output_file = last + '_temp.json'
    image_info = get_image_info(directory)
    save_to_json(image_info, output_file)
    print(f"Image information saved to {output_file}")