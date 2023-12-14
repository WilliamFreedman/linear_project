# code adapted from Khoi Pham at:
# https://vkhoi.github.io/notes/discrete-cosine-transform-in-image-compression


from scipy import fftpack
import numpy as np
import matplotlib.pylab as plt
import cv2
import math
from PIL import Image
from laplacian_blur_degree import *
import os

#step: perform dct on image
def get_2d_dct(im, use_scipy=False):
    # Flag to indicate whether to use the SciPy package.
    if use_scipy:
        return fftpack.dct(fftpack.dct(im.T, norm='ortho').T, norm='ortho')
    
    # Get the signal's length.
    N = im.shape[0]
    
    # Compute the constant scaling factor alpha.
    a = [(1/N)**0.5 if k == 0 else (2/N)**0.5 for k in range(N)]
    
    # Create matrix C that holds the basis cosine vectors.
    C = np.zeros([N, N])
    for k in range(N):
        for n in range(N):
            C[k][n] = a[k]*math.cos(math.pi*(2*n+1)*k/2/N)
    
    # Transforms the input image onto this basis.
    res = np.dot(C, im)
    res = np.dot(res, C.T)
    
    return res


def get_2d_idct(dct, use_scipy=False):
    # Flag to indicate whether to use the SciPy package.
    if use_scipy:
        return fftpack.idct(fftpack.idct(dct.T, norm='ortho').T, norm='ortho')

    # Get the signal's length.
    N = dct.shape[0]
    
    # Compute the constant scaling factor alpha.
    a = [(1/N)**0.5 if k == 0 else (2/N)**0.5 for k in range(N)]
    
    # Create matrix C that holds the basis cosine vectors.
    C = np.zeros([N, N])
    for k in range(N):
        for n in range(N):
            C[k][n] = a[k]*math.cos(math.pi*(2*n+1)*k/2/N)
    
    # Inverse transform.
    res = np.dot(C.T, dct)
    res = np.dot(res, C)
    
    return res

# Get next cell in the JPEG order.
def get_next_cell(cell, N):
    # r: row index; c: column index; d: direction we are going - 0 if going down, 1 if going up.
    r, c, d = cell
    if r == 0 and c % 2 == 0:
        c += 1
        d = 0
        if c >= N:
            r += 1
            c = N-1
    elif c == 0 and r % 2 == 1:
        r += 1
        d = 1
        if r >= N:
            c += 1
            r = N-1
    else:
        if d == 0:
            r += 1
            c -= 1
        else:
            r -= 1
            c += 1
    return (r, c, d)

# Input image in grayscale.

def compress_image(image_path, output_path, step_num):
    if '.jpg' in image_path:
        image_type = '.jpg'
    else:
        image_type = '.png'
    im = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Get image size. For simplicity, we only work with square image.
    N = im.shape[0]
    M = im.shape[1]

    # Compute the DCT coefficients.
    dct = get_2d_dct(im, use_scipy=True)

    # Starting from 0 coefficient, at each step, we choose 20 more coefficients.
    step = step_num

    # The first cell starts from (0, 0).
    cell = (0, 0, 0)

    # Quantized DCT coefficients.
    # We choose to keep the lowest coefficients which is similar to that of JPEG. 
    quantized_dct = np.zeros([N, M])

    # Index of the image for plotting.
    idx = 0

    # 100 steps.
    for i in range(1):
        idx += 1
        
        # Choose additionally more $step$ coefficients.
        for j in range(step):
            cell = get_next_cell(cell, N)
            quantized_dct[cell[0]][cell[1]] = dct[cell[0]][cell[1]]
        
        # Reconstruct.
        reconstructed = get_2d_idct(quantized_dct, use_scipy=True)

    normalized_image = cv2.normalize(reconstructed, None, 0, 1, cv2.NORM_MINMAX)
    # Create a grayscale colormap
    cmap_gray = (plt.cm.gray(normalized_image) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path , 'output_' + str(step_num) + image_type), cmap_gray)

    return output_path + '/output_' + str(step_num) + image_type

def create_out_images(image_path, output_path, step_num):
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
    n=100
    for i in range(0, step_num, n):

        r = compress_image(image_path, './output_images', i)
        new_size = os.path.getsize(r)
        new_blur_degree = calculate_blur_degree(r)
        compression_ratios.append(new_size/original_size)
        blur_degrees.append(new_blur_degree)
    return (compression_ratios, blur_degrees)

def folder_dct(src_folder,step_num):
    step_input = int(step_num/100)
    blurs = np.zeros(step_input)
    ratios = np.zeros(step_input)
    count = 0
    for filename in os.listdir(src_folder):
        count += 1
        filename = os.path.join(src_folder, filename)
        print(filename)
        r = create_out_images(filename,filename,step_num)
        ratios += r[0]
        blurs += r[1]

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(ratios/count, blurs/count)
    ax1.set_title("Compression ratio vs Blue Degree for DCT")
    ax1.set_xlabel("Compression Ratio")
    ax1.set_ylabel("Blur Degree")

    fig1.savefig('./output_graphs/dct/compression_ratio_vs_blur_degree_grayscale.png')

folder_dct('./src_images', 73900)