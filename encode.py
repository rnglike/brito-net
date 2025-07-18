from PIL import Image
import numpy as np
import os

def prepare(input_folder, output_file):
    # Get list of image file paths in the folder (sorted for consistency)
    image_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    if not image_files:
        print("No images found in folder:", input_folder)
        return

    # Open the first image to get size
    first_image = Image.open(image_files[0]).convert('1')  # Convert to black & white (1-bit)
    width, height = first_image.size
    num_pixels = width * height

    print(f"Image size: {width}x{height} ({num_pixels} pixels)")
    print(f"Found {len(image_files)} images.")

    # Initialize matrix: rows=pixels, cols=images
    pixel_matrix = np.zeros((num_pixels, len(image_files)), dtype=np.uint8)

    # Process each image
    for col, img_path in enumerate(image_files):
        img = Image.open(img_path).convert('1')  # Convert to 1-bit (B&W)
        if img.size != (width, height):
            raise ValueError(f"Image {img_path} size {img.size} doesn't match first image size {width}x{height}")

        # Get pixel data as 1D array: 0 for white, 1 for black
        pixel_data = np.array(img).flatten()
        # Invert: white(255)->0, black(0)->1
        pixel_data = np.where(pixel_data == 0, 1, 0)

        # Assign to column
        pixel_matrix[:, col] = pixel_data

    # Save to txt (each row = pixel, each column = image)
    np.savetxt(output_file, pixel_matrix, fmt='%d', delimiter=' ')

    print(f"Saved matrix to {output_file}")

def write(labels, output_file):
    # Convert labels to string and save
    labels_str = ' '.join(map(str, labels))
    with open(output_file, 'w') as f:
        f.write(labels_str)
    print(f"Saved labels to {output_file}")