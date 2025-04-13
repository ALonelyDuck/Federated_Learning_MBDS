import os

def list_images(directory):
    """Alternative à imutils.paths.list_images"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    
    return image_paths