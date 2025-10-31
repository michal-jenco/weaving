from PIL import Image
import numpy as np

# Load the image
image_path = "/mnt/data/bafkreidzwexupfgi27brkwnv6lpx5uznrwixgmoqynkulcr4pouvjxv3oa.jpg"
img = Image.open(image_path).convert("L")  # Convert to grayscale

# Resize to 64x64
img_resized = img.resize((64, 64))

# Convert to numpy array
img_array = np.array(img_resized)

# Normalize and convert to binary (threshold at 128)
binary_array = (img_array > 128).astype(int)

binary_array
