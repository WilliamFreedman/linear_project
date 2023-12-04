import pywt
import numpy as np

def wavelet_decomposition(image, wavelet='haar', level=1):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def quantize(coefficients, threshold=20):
    quantized_coeffs = [np.round(c / threshold) for c in coefficients]
    return quantized_coeffs

def huffman_encode(data):

    encoded_data = ...
    return encoded_data

def compress(image, wavelet='haar', level=1, threshold=20):
    # Wavelet Decomposition
    coeffs = wavelet_decomposition(image, wavelet, level)

    # Quantization
    quantized_coeffs = quantize(coeffs, threshold)

    # Entropy Coding
    compressed_data = huffman_encode(quantized_coeffs)

    return compressed_data

def decompress(compressed_data):
    # Reverse Huffman encoding
    decoded_coeffs = huffman_decode(compressed_data)

    # Reverse Quantization
    dequantized_coeffs = dequantize(decoded_coeffs)

    # Inverse Wavelet Transform
    reconstructed_image = inverse_wavelet_transform(dequantized_coeffs)

    return reconstructed_image

