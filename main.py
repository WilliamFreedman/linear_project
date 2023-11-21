import os
import sys
import imutils
from svd import *

compression_dict = {"svd":compress_svd}
try:
    compressor = compression_dict[sys.argv[1]]
except KeyError:
    print(sys.argv[1] + ' is not a valid compression algorithm.', file=sys.stderr)
    exit(1)

# Set the input and output file paths
input_image_path = "src_images/tree.jpeg"  # Change this to the path of your input image
output_image_path = "compressed_image.jpeg"  # Change this to the desired output path

original_size = os.path.getsize(input_image_path)

# Set the number of singular values to keep (compression factor)
k_values = [2**i for i in range(20)]  # You can experiment with different values


# Apply SVD compression for each k value
for k in k_values:
    output_path_k = f"compressed_image_k{k}.jpg"
    compressor(input_image_path, output_path_k, k)
    compression_ratio = os.path.getsize(output_path_k) / original_size
    os.rename(output_path_k,"./output_images/" + sys.argv[1] + "/"+output_path_k)
    print(f"Compression with k={k} complete. Compression ratio: {compression_ratio}. Output saved to {output_path_k}")
