import os
import sys
import imutils
from svd import *

compression_dict = {"svd":svd_driver}
try:
    compressor = compression_dict[sys.argv[1]]
except KeyError:
    print(sys.argv[1] + ' is not a valid compression algorithm.', file=sys.stderr)
    exit(1)

# Set the input and output file paths
input_image_path = "src_images/tree.jpeg"  # Change this to the path of your input image
output_image_path = "compressed_image.jpeg"  # Change this to the desired output path

compressor(input_image_path)
