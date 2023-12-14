# takes both jpg and png types and compresses them

from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import os
from laplacian_blur_degree import *
from PIL import Image


def dct2(a):
    return fftpack.dct(fftpack.dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return fftpack.dct(fftpack.dct(a.T, norm='ortho').T, norm='ortho')

def compress_image(image, compression_factor, image_type):
    # Split the image into RGB channel

    b, g, r = cv2.split(image)

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
        ])
    return compressed_image

def dct_compress(image_path, output_path, compression):
    # Load image
    if '.jpg' in image_path:
        image_type = '.jpg'
    else:
        image_type = '.png'
    original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Set compression factor (adjust as needed) -> goes from 0 - 1, with 1 being full loss
    # Compress the image
    compressed_image = compress_image(original_image, compression, image_type)
    cv2.imwrite(os.path.join(output_path , 'rgb_output_' + str(compression) + image_type), compressed_image)

    return output_path + '/rgb_output_' + str(compression) + image_type

def create_out_images(image_path, output_path, granularity):
    compression_ratios = []
    blur_degrees = []
    original_image = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(original_image)

    uncompressed = Image.fromarray(img_array)

    # Save the compressed image
    uncompressed.save("./temp.png")

    original_blur = calculate_blur_degree("./temp.png")
    original_size = os.path.getsize("./temp.png")
    print(original_blur)

    for i in [float(j) / granularity for j in range(0, granularity, 1)]:

        r = dct_compress(image_path, './output_images', i)
        new_size = os.path.getsize(r)
        new_blur_degree = calculate_blur_degree(r)
        compression_ratios.append(new_size/original_size)
        blur_degrees.append(new_blur_degree)
    return (compression_ratios, blur_degrees)

def folder_dct(src_folder,granularity):
    blurs = np.zeros(granularity)
    ratios = np.zeros(granularity)
    count = 0
    for filename in os.listdir(src_folder):
        count += 1
        filename = os.path.join(src_folder, filename)
        print(filename)
        r = create_out_images(filename,filename,granularity)
        ratios += r[0]
        blurs += r[1]

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ratios/count, blurs/count)
    ax1.set_title("Compression ratio vs Blur Degree for DCT")
    ax1.set_xlabel("Compression Ratio")
    ax1.set_ylabel("Blur Degree")

    fig1.savefig('./output_graphs/dct/compression_ratio_vs_blur_degree_rgb.png')

folder_dct('./src_images', 100)