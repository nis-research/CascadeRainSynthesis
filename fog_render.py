import numpy as np
import cv2

from util import get_image_brightness


def render_haze(img, dep, intensity_level=50, beta=None):

    h, w, _ = img.shape

    # Smooth depth map using Gaussian filter
    gaussdep = cv2.GaussianBlur(dep, (0, 0), 5)**1.7

    # Beta value controls the density of the haze
    if beta is None:
        beta = np.random.rand() * 2 + 4.7

    # Transmittance
    tx = np.exp(-beta * gaussdep)

    bright_pixel_ratio = get_image_brightness(img)

    # Compute atmospheric light (A) 
    a_min, a_max = 0.4, 0.95
    a = a_min + 2.1 * bright_pixel_ratio - 1.8 * bright_pixel_ratio**2
    a += (np.random.rand() - 0.5) * 0.1 
    a = np.clip(a, a_min, a_max)

    # Compute haze thickness
    thickness = 0.9 - 2.3 * bright_pixel_ratio + 2.3 * bright_pixel_ratio**2
    thickness *= (intensity_level/100)**1.2

    A = np.ones((h, w, 3)) * a

    trans = np.repeat(tx[:, :, np.newaxis], 3, axis=2) ** thickness

    # Render haze effect
    haze = img * trans + (1 - trans) * A

    return haze, trans, (a, thickness, intensity_level), (1 - trans) * A



def render_ground_haze(img, depth_map, ground_mask, intensity_level=50):
    """ 
    Simulates ground haze based on depth and ground mask.

    """

    h, w, _ = img.shape

    # Smooth the Ground Mask to Create Soft Transitions
    blurred_ground = cv2.GaussianBlur(ground_mask.astype(np.float32), (201, 201), 0)

    blurred_ground = cv2.normalize(blurred_ground, None, 0, 1, cv2.NORM_MINMAX)

    #Blend Ground Mask with Depth Map
    depth_map = depth_map**2.3
    depth_map = np.clip(depth_map, 0, 0.8)
    haze_mask = blurred_ground * depth_map

    # Compute Haze Density
    beta = np.random.uniform(5.0, 6.5)
    transmittance = np.exp(-beta * haze_mask)

    # Adjust Intensity Level
    transmittance = transmittance ** (intensity_level / 100)

    A = np.ones((h, w, 3)) * 0.8

    transmittance = np.repeat(transmittance[:, :, np.newaxis], 3, axis=2)  # Match RGB channels
    haze = img * transmittance + (1 - transmittance) * A

    return haze, transmittance, (1 - transmittance) * A
