from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from laplacian_blur_degree import *

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

def create_svd_graphs(k_values,compression_ratios,blur_degrees):
    plt.xscale("log", base=2)
    plt.plot(k_values, compression_ratios)
    plt.title("Compression ratio vs compression factor k")
    plt.xlabel("SVD k value")
    plt.ylabel("Compression ratio")
    plt.savefig('./output_graphs/svd_r_v_k.png')

    plt.clf()

    plt.xscale("log", base=2)
    plt.plot(k_values, blur_degrees)
    plt.title("Blur degree vs compression factor k")
    plt.xlabel("SVD k value")
    plt.ylabel("Blur degree")
    plt.savefig('./output_graphs/svd_b_v_k.png')

#applies svd to given image for k values 2-2^20
def svd_driver(image_path):
    # Set the number of singular values to keep (compression factor)
    original_size = os.path.getsize(image_path)
    k_values = [2 ** i for i in range(10)]  # You can experiment with different values

    compression_ratios = []
    blur_degrees = []

    # Apply SVD compression for each k value
    for k in k_values:
        output_path_k = f"compressed_image_k{k}.jpg"
        compress_svd(image_path, output_path_k, k)
        compression_ratio = os.path.getsize(output_path_k) / original_size
        compression_ratios.append(compression_ratio)
        blur_degrees.append(calculate_blur_degree(output_path_k))
        os.rename(output_path_k, "./output_images/svd/" + output_path_k)
        print(
            f"Compression with k={k} complete. Compression ratio: {compression_ratio}. Output saved to {output_path_k}")

    create_svd_graphs(k_values,compression_ratios,blur_degrees)



svd_driver("src_images/tree.png")

