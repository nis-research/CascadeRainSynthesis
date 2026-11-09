import numpy as np

def get_image_brightness(img):
    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    threshold = 0.7
    bright_pixel_ratio = np.sum(luminance > threshold) / luminance.size

    bright_pixel_ratio = min(bright_pixel_ratio, 0.5)
    return bright_pixel_ratio