import os
import numpy as np
import cv2  # Import OpenCV

def find_image_path(base_name, images_dir):
    """
    Search for an image with base_name in images_dir that has a .jpg, .jpeg, or .png extension.
    Returns the full path if found, else None.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        candidate = os.path.join(images_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def process_annotation_file(label_path, images_dir, sizes, color_mode, output_dir):
    """
    Process a single annotation file and extract patches based on the color mode.
    Handles images with .jpg, .jpeg, or .png extensions automatically.

    Args:
        label_path (str): Path to the annotation file.
        images_dir (str): Directory containing the images.
        sizes (int): Patch size (e.g., 8, 16, 32).
        color_mode (str): Color mode ("RGB", "LAB", "H").
        output_dir (str): Directory to save cropped image patches.

    Returns:
        list: List of rows to be written to the CSV.
    """
    # Derive the base name from the label file (e.g., "image1" from "image1.txt")
    base_name = os.path.splitext(os.path.basename(label_path))[0]
    
    # Find the image path with any supported extension
    image_path = find_image_path(base_name, images_dir)
    if image_path is None:
        print(f"No image found for {label_path}. Skipping...")
        return []

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read the image using OpenCV (BGR), then convert to RGB for consistency
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image at {image_path}. Skipping...")
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img.shape
        
        # Determine which color space to use
        if color_mode == "RGB":
            color_img = img
            channels = ['R', 'G', 'B']
        elif color_mode == "LAB":
            color_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            channels = ['L', 'A', 'B']
        elif color_mode == "H":
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            color_img = hsv_img[:, :, 0]  # Extract only the Hue channel
            channels = ['H']
        else:
            raise ValueError("color_mode must be 'RGB', 'LAB', or 'H'")
        
        # Read all lines from the label file
        with open(label_path, 'r') as f:
            lines = f.read().strip().split('\n')
        
        half_size = sizes // 2
        results = []
        
        for idx, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue

            # Each line typically: class_id x_center y_center ...
            parts = line.split()
            class_id = int(parts[0])
            cx = int(round(float(parts[1]) * img_w))
            cy = int(round(float(parts[2]) * img_h))
            
            # Calculate patch boundaries
            x_start, x_end = max(0, cx - half_size), min(img_w, cx + half_size)
            y_start, y_end = max(0, cy - half_size), min(img_h, cy + half_size)
            
            # Extract the patch
            if color_mode == "H":
                patch = color_img[y_start:y_end, x_start:x_end]
            else:
                patch = color_img[y_start:y_end, x_start:x_end, :]
            
            # Zero-pad if needed
            pad_height = sizes - patch.shape[0]
            pad_width = sizes - patch.shape[1]
            if pad_height > 0 or pad_width > 0:
                if color_mode == "H":
                    patch = np.pad(
                        patch,
                        ((0, pad_height), (0, pad_width)),
                        mode='constant',
                        constant_values=0
                    )
                else:
                    patch = np.pad(
                        patch,
                        ((0, pad_height), (0, pad_width), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
            
            # Save the cropped patch (convert back to BGR if multi-channel)
            patch_output_path = os.path.join(output_dir, f"{base_name}_{class_id}_{idx}.png")
            if color_mode == "H":
                cv2.imwrite(patch_output_path, patch)
            else:
                cv2.imwrite(patch_output_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            
            # Flatten the patch and build the CSV row
            # The image name in the CSV can be the actual file name (with extension)
            image_filename = os.path.basename(image_path)  # e.g., "image1.jpg" or "image1.png"
            
            if color_mode == "H":
                row = [image_filename, class_id] + patch.flatten().tolist()
            else:
                flattened = [patch[:, :, c].flatten().tolist() for c in range(patch.shape[2])]
                row = [image_filename, class_id] + sum(flattened, [])
            
            results.append(row)
        
        return results
    except Exception as e:
        print(f"Error processing {label_path}: {e}")
        return []
