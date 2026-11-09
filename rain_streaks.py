import numpy as np
import cv2

from util import get_image_brightness

def render_rain(img, density, streaks_length, transparency, blurriness, thickness, dep_mask=None, rain_angle=80):
    h, w, c = img.shape
    theta = rain_angle + (np.random.rand() - 0.5) * 10  # Fixed rain streak angle (slightly random)

    # Generate noise mask
    normalized_density = np.clip(density / 100, 0, 1)
    noise = (np.random.rand(h, w) < normalized_density * 0.1).astype(np.float32)

    # Apply thickness by spreading horizontally
    if thickness > 1:
        kernel = np.ones((1, int(thickness)), dtype=np.uint8)
        noise = cv2.dilate(noise, kernel)

    noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)

    # Apply depth mask
    if dep_mask is not None:
        noise[dep_mask > 0] = 0

    # Create motion blur
    size = max(1, int(streaks_length))
    motion_blur_kernel = np.zeros((size, size), dtype=np.float32)
    cv2.line(motion_blur_kernel, (0, size // 2), (size - 1, size // 2), 1, thickness=1)

    M = cv2.getRotationMatrix2D((size // 2, size // 2), theta, 1)  # Rotate around center
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (size, size))

    motion_blur_kernel /= motion_blur_kernel.sum()

    rain_mask = cv2.filter2D(noise[:, :, 0], -1, motion_blur_kernel)
    rain_mask = np.repeat(rain_mask[:, :, np.newaxis], 3, axis=2)

    if blurriness > 0:
        rain_mask = cv2.GaussianBlur(rain_mask, (5, 5), sigmaX=blurriness)

    # Compute transparency factor
    mid_trans_scale = 1.5
    transparency_factor = ((100 - transparency) / 50) * mid_trans_scale
    if transparency < 50:
        transparency_factor = mid_trans_scale + (50 - transparency) * ((10 - mid_trans_scale) / 50)

    rain_mask *= transparency_factor

    # Add rain effect to the image
    image_rain = np.clip(img + rain_mask, 0, 1)

    return image_rain, rain_mask




def set_streak_lengths(params, intensity_level):
    """Interpolates params[i][1] based on intensity_level (10 to 100)."""
    intensity_level = max(10, min(100, intensity_level))
    
    # Define max and min values for params[i][1]
    min_values = [50, 45, 40, 35, 30, 25, 20, 15, 10, 4]
    max_values = [120, 100, 90, 80, 65, 50, 35, 25, 10, 4]

    interpolated_values = [
        min_val + (max_val - min_val) * ((intensity_level - 10) / (100 - 10))
        for min_val, max_val in zip(min_values, max_values)
    ]
    
    # Update params[i][1] with interpolated values
    for i in range(len(params)):
        params[i][1] = interpolated_values[i]

    # Add random offset
    rand_factor = np.random.uniform(0, 1)
    for i in range(len(params)):
        params[i][1] += rand_factor * (max_values[i]-min_values[i])/5.
    
    return params




def set_streak_transparency(params, brightness_level):
    """Interpolates params[i][2] based on brightness_level"""
    brightness_level = max(0, min(0.35, brightness_level))
    
    # Define max and min values for params[i][1]
    min_values = [20, 35, 25, 50, 40, 65, 70, 75, 80, 90]
    max_values = [45, 45, 55, 55, 65, 70, 75, 80, 80, 95]

    interpolated_values = [
        min_val + (max_val - min_val) * ((0.35 - brightness_level) / 0.35)
        for min_val, max_val in zip(min_values, max_values)
    ]
    
    # Update params[i][1] with interpolated values
    for i in range(len(params)):
        params[i][2] = interpolated_values[i]
    
    
    return params




def render_depth_based_rain(img, depth_map, intensity_level=50, num_layers=None, base_name=None):

    base_rain_angle = np.random.uniform(55, 125)

    if num_layers is None:

        params = [
            [0.1, 80, 45, 0.35, 6, 1],
            [0.8, 70, 45, 0.35, 1, 1],
            #
            [0.15, 60, 55, 0.5, 5, 0.9],
            [1.25, 55, 55, 0.5, 1, 0.8],
            #
            [0.2, 40, 65, 0.65, 4, 0.7],
            [2, 40, 70, 0.65, 1, 0.6],
            #
            [0.4, 25, 75, 0.8, 3, 0.5],
            [3, 15, 80, 0.8, 1, 0.4],
            #
            [3, 10, 80, 1, 1, 0.3],
            [4, 4, 95, 1, 1, 0.2],
        ]


        default_intensity = 80
        for i in range(len(params)):
            params[i][0] *= intensity_level/default_intensity

        params = set_streak_lengths(params, intensity_level)
        params = set_streak_transparency(params, get_image_brightness(img))


        num_layers = len(params)
    else:
        start_params = [10, 30, 50, 0.5, 1, 1]
        end_params = [30, 4, 90, 0.7, 1, 0.1]
        params = [start_params * (1 - t) + end_params * t for t in np.linspace(0, 1, num_layers)]
    
    if len(depth_map.shape) == 2:
        depth_map = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2)
    depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
    
    depth_levels_mask = np.zeros((depth_map.shape[0], depth_map.shape[1], depth_map.shape[2], num_layers), dtype=bool)
    depth_levels_mask = [depth_map > params[i][5] for i in range(num_layers)]
    
    final_rain_mask = np.zeros_like(img, dtype=np.float32)
    for i in range(num_layers):
        density, streaks_length, transparency, blurriness, thickness = params[i][:5]

        _, rain_mask = render_rain(img, density, streaks_length, transparency, blurriness, thickness, depth_levels_mask[i], base_rain_angle)
        if i > 0:
            rain_mask[depth_levels_mask[i-1]] = 0
        
        mask_path = f"RainMasks/{base_name}_rainmask_{i}.png"
        cv2.imwrite(mask_path, (rain_mask * 255).astype(np.uint8))

        final_rain_mask += rain_mask
    
    final_rain_mask = np.clip(final_rain_mask, 0, 1)
    mask_path = f"RainMasks/{base_name}_rainmask_final.png"
    cv2.imwrite(mask_path, (final_rain_mask * 255).astype(np.uint8))

    final_image_rain = np.clip(img + final_rain_mask, 0, 1)
    
    return final_image_rain, final_rain_mask
