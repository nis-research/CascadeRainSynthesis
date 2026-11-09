import numpy as np
import cv2
import os
from tqdm import tqdm

from fog_render import render_haze, render_ground_haze
from puddles import create_puddles
from rain_streaks import render_depth_based_rain


def batch_render_weather(intensity_level=None,
                   input_dir="Input", output_dir="Output",
                   depth_dir = "DepthMaps", ground_dir = "GroundMasks",
                   save_base=True, save_wet=True, save_puddles=True,
                   beta=None, res_value=None, persistence_value=None, reflection_value=None, 
                   ):
                   

    for filename in tqdm(os.listdir(input_dir), desc="Rendering Weather"): 
    
        rain_intensity = intensity_level if intensity_level is not None else np.random.uniform(10, 75)
        render_weather(
            image_base_name=filename,
            intensity_level=rain_intensity,
            input_dir=input_dir,
            output_dir=output_dir,
            depth_dir = depth_dir,
            ground_dir = ground_dir,
            save_base=save_base,
            save_wet=save_wet,
            save_puddles=save_puddles,
            beta=beta,
            res_value=res_value,
            persistence_value=persistence_value,
            reflection_value=reflection_value,
            )


def apply_streaks_and_haze(img, depth_map, ground_mask, rain_mask, rain_intensity_level, beta=None):
    # Generate haze
    haze, trans, atm, _ = render_haze(img, depth_map, rain_intensity_level, beta=beta)

    # Generate ground haze
    hazeG, trans, atm = render_ground_haze(haze, depth_map, ground_mask, rain_intensity_level)

    # Final image with rain and haze
    final_img = hazeG + rain_mask
    final_img = np.clip(final_img, 0, 1)

    final_img_uint8 = (final_img * 255).astype(np.uint8)
    final_img_uint8 = final_img_uint8[..., ::-1]

    return final_img_uint8


def render_weather(image_base_name, intensity_level=50,
                   input_dir="Input", output_dir="Output",
                   depth_dir = "DepthMaps", ground_dir = "GroundMasks",
                   save_base=True, save_wet=True, save_puddles=True,
                   beta=None, res_value=None, persistence_value=None,
                   reflection_value=None):
    
    base_name, _ = os.path.splitext(image_base_name)

    image_gt = os.path.join(input_dir, image_base_name)    
    image_depth = os.path.join(depth_dir, f"{base_name}_depth.png")
    image_ground = os.path.join(ground_dir, f"{base_name}_ground.png")

    semantic_mask_gt = f"SS_GT/{base_name.replace('leftImg8bit', 'gtFine_color')}.png"

    img = cv2.imread(image_gt)
    img = img[..., ::-1]

    depth_map = cv2.imread(image_depth, cv2.IMREAD_GRAYSCALE) / 255.0
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]))

    ground_mask = cv2.imread(image_ground)
    ground_mask = cv2.resize(ground_mask, (img.shape[1], img.shape[0]))
    ground_mask_gray = cv2.imread(image_ground, cv2.IMREAD_GRAYSCALE) / 255.0


    # === PUDDLES / WETNESS GENERATION ===
    semantic_mask = None
    if os.path.exists(semantic_mask_gt):
        semantic_mask = cv2.imread(semantic_mask_gt)

    mask_input = semantic_mask if semantic_mask is not None else ground_mask

    img_wet = None
    img_puddles = None

    if save_puddles:
        img_puddles, img_wet = create_puddles(img.copy(), mask_input, rain_intensity=intensity_level, need_puddles=True, res_value=res_value, persistence_value=persistence_value, reflection_value=reflection_value)
        img_puddles = img_puddles / 255.0
        img_wet = img_wet / 255.0
    elif save_wet:
        _, img_wet = create_puddles(img.copy(), mask_input, rain_intensity=intensity_level, need_puddles=False, res_value=res_value, persistence_value=persistence_value, reflection_value=reflection_value)
        img_wet = img_wet / 255.0
    
    img = img / 255.0


    # === RAIN MASK GENERATION ===
    img_for_rain = img_puddles if img_puddles is not None else img
    rainy_img, rain_mask = render_depth_based_rain(img_for_rain, depth_map, intensity_level, base_name=base_name)

    depth_map_haze = 1.0 / (depth_map + 0.1)
    depth_map_haze /= np.max(depth_map_haze)

    if save_base:
        result = apply_streaks_and_haze(img, depth_map_haze, ground_mask_gray, rain_mask, intensity_level, beta=beta)
        cv2.imwrite(os.path.join(output_dir, "Base", f"{base_name}_rain.png"), result)

    if save_wet:
        result_wet = apply_streaks_and_haze(img_wet, depth_map_haze, ground_mask_gray, rain_mask, intensity_level, beta=beta)
        cv2.imwrite(os.path.join(output_dir, "Wet", f"{base_name}_rain_w.png"), result_wet)

    if save_puddles:
        result_puddles = apply_streaks_and_haze(img_puddles, depth_map_haze, ground_mask_gray, rain_mask, intensity_level, beta=beta)
        cv2.imwrite(os.path.join(output_dir, "Puddles", f"{base_name}_rain_wp.png"), result_puddles)
    