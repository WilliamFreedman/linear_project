# takes both jpg and png types and compresses them

import numpy as np
import cv2
from PIL import Image

def dct2(a):
    return np.fft.fft2(a)

def idct2(a):
    return np.fft.ifft2(a)

def compress_image(image, compression_factor, image_type):
    # Split the image into RGB channels
    if 'jpg' == image_type:
        b, g, r = cv2.split(image)
    else:
        b, g, r, a = cv2.split(image)

    # Apply DCT to each channel (excluding alpha)
    red_dct = dct2(r)
    green_dct = dct2(g)
    blue_dct = dct2(b)

    # Set high-frequency components to zero based on compression factor
    rows, cols = image.shape[0], image.shape[1]
    mask = np.ones_like(red_dct)
    mask[:int(rows * compression_factor), :] = 0
    mask[:, :int(cols * compression_factor)] = 0

    red_dct *= mask
    green_dct *= mask
    blue_dct *= mask

    # Reconstruct the compressed image
    compressed_red_channel = idct2(red_dct).real
    compressed_green_channel = idct2(green_dct).real
    compressed_blue_channel = idct2(blue_dct).real

    # Stack the compressed channels to get the final image
    if 'jpg' == image_type:
        compressed_image = cv2.merge([
        compressed_blue_channel.astype(np.uint8),
        compressed_green_channel.astype(np.uint8),
        compressed_red_channel.astype(np.uint8),
    ])
    else:
        compressed_image = cv2.merge([
            compressed_blue_channel.astype(np.uint8),
            compressed_green_channel.astype(np.uint8),
            compressed_red_channel.astype(np.uint8),
            a  # Include the alpha channel without modification
        ])
    return compressed_image

def dct_compress(image_path, output_path, compression):
    # Load image
    if '.jpg' in image_path:
        image_type = 'jpg'
    else:
        image_type = 'png'
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Set compression factor (adjust as needed) -> goes from 0 - 1, with 1 being full loss
    # Compress the image
    compressed_image = compress_image(original_image, compression, image_type)
    save_image = Image.fromarray(compressed_image)
    save_image.save(output_path)

def create_out_images(image_path, output_path):
    for i in [float(j) / 10000 for j in range(0, 10000, 1)]:
        dct_compress(image_path, output_path, i)