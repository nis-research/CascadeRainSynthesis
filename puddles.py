import numpy as np
import cv2

from ground_mask import extract_masks, shrink_mask
from util import get_image_brightness

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)



def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.7):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise



def generate_random_fractal_noise(res_value=None, persistence_value=None):
    if res_value is None:
        res = (8, 8) if np.random.rand() > 0.5 else (4, 4)
    else:
        res = (res_value, res_value)

    if persistence_value is None:
        persistence = np.random.uniform(0.4, 0.45)
    else:
        persistence = persistence_value


    noise = generate_fractal_noise_2d(
        (256, 256),
        res,
        5,
        persistence,
    )

    noise *= 2
    noise = np.clip(noise, 0, 1)

    return noise



def create_wave_displacement_map(shape, scale=3.0, freq=0.01):
    """Generate sinusoidal wave displacement maps for x and y directions."""
    height, width = shape[:2]

    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)

    dx = (np.sin(yy * freq * 2 * np.pi) + np.cos(xx * freq * 2 * np.pi)) * scale
    dy = (np.cos(yy * freq * 2 * np.pi) + np.sin(xx * freq * 2 * np.pi)) * scale

    return dx.astype(np.float32), dy.astype(np.float32)

def generate_chaotic_displacement(shape, scale=5, blur_ksize=15):
    """Generate chaotic random displacement maps."""
    height, width = shape[:2]
    
    dx = (np.random.rand(height, width) - 0.5) * 2 * scale
    dy = (np.random.rand(height, width) - 0.5) * 2 * scale

    dx = cv2.GaussianBlur(dx, (blur_ksize, blur_ksize), 0)
    dy = cv2.GaussianBlur(dy, (blur_ksize, blur_ksize), 0)

    return dx.astype(np.float32), dy.astype(np.float32)

def apply_displacement_map(image, dx, dy):
    """Apply displacement map to the image."""
    height, width = image.shape[:2]
    
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)
    
    distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return distorted



def apply_vertical_motion_blur(image, kernel_size=15):
    """Apply vertical motion blur to an image."""

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[:, kernel_size // 2] = np.ones(kernel_size, dtype=np.float32)
    kernel /= kernel_size  # Normalize the kernel

    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred



def generate_puddle_map(shape, intensity=0.5, res_value=None, persistence_value=None):
    """Generate a black-and-white puddle map."""

    puddle_map = generate_random_fractal_noise(
        res_value=res_value,
        persistence_value=persistence_value
        )

    puddle_map = cv2.normalize(puddle_map, None, 0, 255, cv2.NORM_MINMAX)
    puddle_map = puddle_map.astype(np.uint8)
    return puddle_map



def find_horizon(mask, x_range):
    """Find the highest pixel where there is ground within the given x-range."""
    h, _ = mask.shape
    for y in range(h):
        if np.any(mask[y, x_range] > 0):
            return max(y, h // 2)  # Ensure horizon is not above the center
    return h // 2  # Default to middle if no ground found



def apply_perspective_transform(image, puddle_map, ground_mask, surface_mask):
    """Warp the puddle map to match ground perspective."""
    h, w = image.shape[:2]
    horizon_y = find_horizon(ground_mask, np.arange(w))
    
    # Define source and destination points for perspective transform
    src_h, src_w = puddle_map.shape[:2]
    src_pts = np.float32([[0, 0], [src_w, 0], [0, src_h], [src_w, src_h]])
    dst_pts = np.float32([[0, horizon_y], [w, horizon_y], [w * -0.2, h], [w * 1.2, h]])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_puddle_map = cv2.warpPerspective(puddle_map, M, (w, h))

    # Apply ground mask to remove puddles from non-ground areas
    surface_mask_blurred = cv2.GaussianBlur(surface_mask.astype(np.float32), (201, 201), 0) / 255.0
    warped_puddle_map = warped_puddle_map * surface_mask_blurred
    
    return warped_puddle_map




def generate_water_splash_texture(shape, num_splashes=10000, max_radius=3, brightness=15, kernel_size=3):
    """
    Generate a texture with many small random splashes that vary in shape and sharpness.
    """
    texture = np.zeros(shape, dtype=np.float32)
    
    for _ in range(num_splashes):
        x, y = np.random.randint(0, shape[1]), np.random.randint(0, shape[0])
        radius = np.random.randint(1, max_radius)
        axes = (radius, np.random.randint(1, max_radius + 1))
        angle = np.random.randint(0, 360)
        cv2.ellipse(texture, (x, y), axes, angle, 0, 360, brightness, -1)
    
    texture = cv2.GaussianBlur(texture, (kernel_size, kernel_size), 0)  # Blur some splashes
    return texture




def make_ripples(image, puddles, num_splashes=10000, max_radius=3, brightness=15, kernel_size=3):
    # Generate ripple noise using splash texture
    h, w, _ = image.shape
    noise = generate_water_splash_texture((h, w), num_splashes, max_radius, brightness, kernel_size)
    noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
    puddles = np.clip(puddles + noise, 0, 255) 
    return puddles



def create_puddle_edge_mask(puddle_map, min_width=3, max_width=10, noise_blur_size=101, random_seed=None):
    """
    Create a mask for puddle edges with random smooth width variation.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    puddle_binary = (puddle_map > 100).astype(np.uint8)

    base_dilate = cv2.dilate(puddle_binary, np.ones((max_width*2+1, max_width*2+1), np.uint8), iterations=1)

    random_noise = np.random.uniform(0, 1, size=puddle_binary.shape).astype(np.float32)

    if noise_blur_size > 0:
        random_noise = cv2.GaussianBlur(random_noise, (noise_blur_size, noise_blur_size), 0)

    random_noise = (random_noise - random_noise.min()) / (random_noise.max() - random_noise.min())

    random_width_map = min_width + (max_width - min_width) * random_noise

    variable_dilate = np.zeros_like(puddle_binary, dtype=np.uint8)

    for width in range(min_width, max_width + 1):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*width+1, 2*width+1))
        dilated = cv2.dilate(puddle_binary, kernel, iterations=1)
        mask = (random_width_map >= width) & (random_width_map < width + 1)
        variable_dilate[mask] = dilated[mask]

    puddle_edge = np.clip(variable_dilate - puddle_binary, 0, 1)

    return (puddle_edge * 255).astype(np.uint8)




def apply_wetness_effect(image, edge_mask, strength=1, mask_blur_size=31, noise_blur_size=301, random_seed=None):
    """
    Apply a random wetness effect around puddles on an image, with large smooth bright/dark areas.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    mask = (edge_mask.astype(np.float32) / 255.0)

    # Blur the edge mask to create smooth transition
    if mask_blur_size > 0:
        mask = cv2.GaussianBlur(mask, (mask_blur_size, mask_blur_size), 0)

    mask = np.expand_dims(mask, axis=-1)

    # Generate random noise map
    random_noise = np.random.uniform(-1, 1, size=(mask.shape[0], mask.shape[1])).astype(np.float32)

    # Heavily blur the noise to make large smooth areas
    if noise_blur_size > 0:
        random_noise = cv2.GaussianBlur(random_noise, (noise_blur_size, noise_blur_size), 0)

    min_val, max_val = random_noise.min(), random_noise.max()
    if max_val - min_val > 0:
        random_noise = 2.0 * (random_noise - min_val) / (max_val - min_val) - 1.0

    random_noise = np.expand_dims(random_noise, axis=-1)

    img = image.astype(np.float32) / 255.0

    # Compute brightness adjustment: random_noise * strength
    adjust = 1.04 + random_noise * strength

    img_adjusted = img * (1 - mask) + img * adjust * mask

    img_adjusted = np.clip(img_adjusted, 0, 1)
    return (img_adjusted * 255).astype(np.uint8)



def create_reflection(image, puddle_map, ground_mask, rain_intensity=50, reflection_value=None):
    """Create reflections in puddles using vertical strip-based mirroring."""
    
    h, w, _ = image.shape
    num_strips = 100
    strip_width = w // num_strips
    output = image.copy().astype(np.float32)
    
    for i in range(num_strips):
        x_start = i * strip_width
        x_end = min((i + 1) * strip_width, w)
        
        # Find horizon for this strip
        horizon_y = find_horizon(ground_mask, np.arange(x_start, x_end))
        
        # Extract strip above the horizon
        strip = image[:horizon_y, x_start:x_end]
        
        # Flip and place it below horizon
        flipped_strip = cv2.flip(strip, 0)
        
        # Define region to place the reflection
        reflection_height = min(flipped_strip.shape[0], h - horizon_y)
        output[horizon_y:horizon_y + reflection_height, x_start:x_end] = flipped_strip[:reflection_height]

    # Apply blur to reflections
    output = cv2.GaussianBlur(output, (15, 15), 15)

    reflection = output

    dx, dy = create_wave_displacement_map(reflection.shape, scale=np.random.uniform(2.5, 3.5), freq=np.random.uniform(0.005, 0.01))
    distorted_reflection = apply_displacement_map(reflection, dx, dy)

    dx, dy = generate_chaotic_displacement(reflection.shape, scale=np.random.uniform(5, 7), blur_ksize=5)
    distorted_reflection = apply_displacement_map(reflection, dx, dy)

    output = distorted_reflection

    alpha_mask = puddle_map.astype(np.float32) / 255.0

    # Adjust transparency/reflectivity depending on scene brightness
    reflection_coeff = 0.75 if reflection_value is None else reflection_value

    reflection_coeff *= (get_image_brightness(image)/0.5)**0.4
    reflection_coeff *= (1.0 - rain_intensity/100.0)**0.4
    alpha_mask *= reflection_coeff

    brightness_mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    brightness_mask = (1 - brightness_mask)[:, :, None]
    output *= brightness_mask

    # Generate ripple noise
    output = make_ripples(image, output, num_splashes=3000, max_radius=2, brightness=14, kernel_size=1)
    output = make_ripples(image, output, num_splashes=4000, max_radius=3, brightness=10, kernel_size=3)
    output = make_ripples(image, output, num_splashes=4000, max_radius=2, brightness=12, kernel_size=3)
    output = make_ripples(image, output, num_splashes=3000, max_radius=3, brightness=13, kernel_size=3)
    
    # Blend original image and blurred reflection based on alpha_mask
    final_output = image.astype(np.float32) * (1 - alpha_mask[..., None]) + output * alpha_mask[..., None]
    final_output = np.clip(final_output, 0, 255).astype(np.uint8)

    puddle_edge_mask = create_puddle_edge_mask(
        puddle_map,
        min_width=0, 
        max_width=20,
        noise_blur_size=21,
    )
    final_output = apply_wetness_effect(final_output, puddle_edge_mask, strength=0.5, mask_blur_size=45)
    
    return final_output



def create_glittering_reflection(image, ground_mask):
    """Create a very blurred and transparent reflection across the entire ground area, making bright reflections visible and dark areas darker."""
    h, w, _ = image.shape
    num_strips = 20
    strip_width = w // num_strips
    glittering_reflection = np.zeros_like(image, dtype=np.float32)
    
    for i in range(num_strips):
        x_start = i * strip_width
        x_end = min((i + 1) * strip_width, w)
        
        # Find horizon for this strip
        horizon_y = find_horizon(ground_mask, np.arange(x_start, x_end))
        
        # Extract strip above the horizon
        strip = image[:horizon_y, x_start:x_end]
        
        # Flip and place it below horizon
        flipped_strip = cv2.flip(strip, 0)
        reflection_height = min(flipped_strip.shape[0], h - horizon_y)
        glittering_reflection[horizon_y:horizon_y + reflection_height, x_start:x_end] = flipped_strip[:reflection_height]
    

    glittering_reflection = apply_vertical_motion_blur(glittering_reflection, kernel_size=200)
    
    glittering_reflection = cv2.GaussianBlur(glittering_reflection, (25, 25), 33)

    ground_mask_blurred = cv2.GaussianBlur(ground_mask.astype(np.float32), (201, 201), 0) / 255.0
    
    # Adjust brightness: enhance bright reflections and darken dark areas
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    reflection_intensity = (image_gray * 5 - 1).clip(0, 1)[:, :, None]
    glittering_reflection = glittering_reflection * reflection_intensity
    
    # Reduce opacity only in ground areas using blurred mask
    alpha_glitter = ground_mask_blurred * 0.75
    output = image.astype(np.float32) * (1 - alpha_glitter[..., None]) + glittering_reflection * alpha_glitter[..., None]
    output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output



def create_puddles(image, semantic_map, ground_mask=None, rain_intensity=50, need_puddles=True, res_value=None, persistence_value=None, reflection_value=None):
    """
    Returns:
        - wet_image if need_puddles is False
        - (puddle_image, wet_image) if need_puddles is True
    """
    if semantic_map is None:
        ground_mask *= 255
        glittered_image = create_glittering_reflection(image, ground_mask)
        return (None, glittered_image) if not need_puddles else (glittered_image, glittered_image)

    road_mask, sidewalk_mask, parking_mask, ground_ct_mask = extract_masks(semantic_map)
    ground_mask = road_mask + sidewalk_mask + parking_mask + ground_ct_mask

    road_mask = shrink_mask(road_mask, kernel_size=7, iterations=7)
    sidewalk_mask = shrink_mask(sidewalk_mask, kernel_size=7, iterations=7)
    parking_mask = shrink_mask(parking_mask, kernel_size=3, iterations=3)
    ground_ct_mask = shrink_mask(ground_ct_mask, kernel_size=4, iterations=4)

    glittered_image = create_glittering_reflection(image, ground_mask)

    if not need_puddles:
        return None, glittered_image

    # Puddles-specific processing
    puddle_map1 = generate_puddle_map(image.shape, 0.01, res_value=res_value, persistence_value=persistence_value)
    warped_puddle_map1 = apply_perspective_transform(glittered_image, puddle_map1, ground_mask, road_mask)

    puddle_map2 = generate_puddle_map(image.shape, 0.01, res_value=res_value, persistence_value=persistence_value)
    warped_puddle_map2 = apply_perspective_transform(glittered_image, puddle_map2, ground_mask, sidewalk_mask)

    puddle_map3 = generate_puddle_map(image.shape, 0.01, res_value=res_value, persistence_value=persistence_value)
    warped_puddle_map3 = apply_perspective_transform(glittered_image, puddle_map3, ground_mask, parking_mask)

    puddle_map4 = generate_puddle_map(image.shape, 0.01, res_value=res_value, persistence_value=persistence_value)
    warped_puddle_map4 = apply_perspective_transform(glittered_image, puddle_map4, ground_mask, ground_ct_mask)

    warped_puddle_map = warped_puddle_map1 + warped_puddle_map2 + warped_puddle_map3 + warped_puddle_map4

    puddle_result = create_reflection(glittered_image, warped_puddle_map, ground_mask, rain_intensity, reflection_value=reflection_value)

    return puddle_result, glittered_image