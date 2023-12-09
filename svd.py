from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from laplacian_blur_degree import *

#clarifying note: according to Google, Descartes invented linear algebra, which is why he has the honor of being a sample image

def compress_svd(image_path, output_path, k):
    #k is positively correlated with the amount of information retained from the original matrix
    # Load the image using Pillow
    original_image = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(original_image)

    # Separate the image into its three color channels (R, G, B)
    red, green, blue = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # Apply SVD compression to each color channel
    u_red, s_red, vh_red = np.linalg.svd(red, full_matrices=False)
    u_green, s_green, vh_green = np.linalg.svd(green, full_matrices=False)
    u_blue, s_blue, vh_blue = np.linalg.svd(blue, full_matrices=False)

    # Keep only the first k singular values for compression
    compressed_red = np.dot(u_red[:, :k], np.dot(np.diag(s_red[:k]), vh_red[:k, :]))
    compressed_green = np.dot(u_green[:, :k], np.dot(np.diag(s_green[:k]), vh_green[:k, :]))
    compressed_blue = np.dot(u_blue[:, :k], np.dot(np.diag(s_blue[:k]), vh_blue[:k, :]))

    # Stack the compressed channels to reconstruct the compressed image
    compressed_img_array = np.stack([compressed_red, compressed_green, compressed_blue], axis=-1).clip(0, 255).astype(
        np.uint8)

    # Create a new Pillow image from the compressed array
    compressed_image = Image.fromarray(compressed_img_array)

    # Save the compressed image
    compressed_image.save(output_path)

def create_svd_graphs(k_values, compression_ratios, blur_degrees):
    # Create subplots for compression ratios and blur degrees
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax1.set_xscale("log", base=2)
    ax1.plot(k_values, compression_ratios)
    ax1.set_title("Compression ratio vs compression factor k")
    ax1.set_xlabel("SVD k value")
    ax1.set_ylabel("Compression ratio")

    ax2.set_xscale("log", base=2)
    ax2.plot(k_values, blur_degrees)
    ax2.set_title("Blur degree vs compression factor k")
    ax2.set_xlabel("SVD k value")
    ax2.set_ylabel("Blur degree")

    ax3.set_xscale("log", base=2)
    ax3.plot(compression_ratios, blur_degrees)
    ax3.set_title("Compression Ratio vs. Blur Degree")
    ax3.set_xlabel("Compression Ratio")
    ax3.set_ylabel("Blur degree")

    plt.tight_layout()

    # Save the graphs separately
    fig1.savefig('./output_graphs/svd/compression_ratio_vs_k.png')
    fig2.savefig('./output_graphs/svd/blur_degree_vs_k.png')
    fig3.savefig('./output_graphs/svd/compression_ratio_vs_blur_degree.png')

def process_images_in_folder(folder_path,k_vals):
    count = 0
    compression_ratios = np.zeros(len(k_vals))
    blur_degrees = np.zeros(len(k_vals))
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file extensions as needed
            count += 1
            image_path = os.path.join(folder_path, filename)
            r = svd_driver(image_path,k_vals)
            compression_ratios += r[0]
            blur_degrees += r[1]

    create_svd_graphs(k_vals, compression_ratios/count, blur_degrees/count)


def svd_driver(image_path,k_values):
    original_blur = calculate_blur_degree(image_path)
    original_size = os.path.getsize(image_path)

    compression_ratios = []
    blur_degrees = []

    # Apply SVD compression for each k value
    for k in k_values:
        output_path_k = f"compressed_image_k{k}_{os.path.basename(image_path)}"
        compress_svd(image_path, output_path_k, k)
        compression_ratio = os.path.getsize(output_path_k) / original_size
        compression_ratios.append(compression_ratio)
        blur_degrees.append((calculate_blur_degree(output_path_k))/original_blur)
        os.rename(output_path_k, "./output_images/svd/" + output_path_k)
        print(
            f"Compression for {os.path.basename(image_path)} with k={k} complete. Compression ratio: {compression_ratio}. Output saved to {output_path_k}")
    return (compression_ratios,blur_degrees)



#process_images_in_folder("src_images",list(range(24))+list(range(24,2**10,25)))

process_images_in_folder("src_images",list(range(5)))

