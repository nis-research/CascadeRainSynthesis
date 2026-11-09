import torch
import numpy as np
import cv2
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

from tqdm import tqdm
import os

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ADE20K class list
ADE20K_CLASSES = np.array([
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "window", "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain",
    "plant", "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea",
    "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe",
    "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest",
    "counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool", "pillow", "screen", "stairway", "river",
    "bridge", "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair",
    "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "street light", "booth", "tv", "pole", "land", "bannister",
    "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain",
    "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool", "barrel",
    "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen door", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light",
    "tray", "trash can", "fan", "pier", "crt screen", "plate", "monitor", "bulletin board",
    "shower", "radiator", "glass", "clock", "flag"
])

CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(ADE20K_CLASSES)}

# Class -> color map (according to Cityscapes)
CLASS_COLOR_MAP = {
    "road": (128, 64, 128),
    "sidewalk": (232, 35, 244),
    "parking": (160, 170, 250),
    "floor": (81, 0, 81),
    "earth": (81, 0, 81),
}

# Group of all relevant ground-like classes
SELECTED_CLASSES = {
    "floor", "road", "grass", "sidewalk", "earth", "water", "sea", "field",
    "rock", "sand", "path", "stairs", "runway", "pool", "stairway", "river",
    "hill", "land", "swimming pool", "lake"
}


def create_ground_mask(image_path, ground_mask_path, feature_extractor, model, device):

    print(f"opening img: {image_path}")
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size

    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()} #new to use gpu
    
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    segmentation_map = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()

    segmentation_map_resized = cv2.resize(segmentation_map, orig_size, interpolation=cv2.INTER_NEAREST)

    color_mask = np.zeros((segmentation_map_resized.shape[0], segmentation_map_resized.shape[1], 3), dtype=np.uint8)

    for class_name in SELECTED_CLASSES:
        if class_name not in CLASS_NAME_TO_ID:
            continue
        class_id = CLASS_NAME_TO_ID[class_name]
        if class_name in CLASS_COLOR_MAP:
            color = CLASS_COLOR_MAP[class_name]
        else:
            color = (255, 255, 255)  # White for generic ground

        color_mask[segmentation_map_resized == class_id] = color

    cv2.imwrite(ground_mask_path, color_mask)




def batch_create_ground_masks(input_folder, ground_dir):

    #model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
    #feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
    #segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_name, use_safetensors=True)
    #segformer_model.eval() 
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count()) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Segformer Using device: {device}")
    # test save
    
    local_model_path = "./huggingface/segformer"
    feature_extractor = SegformerImageProcessor.from_pretrained(local_model_path)
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(local_model_path, use_safetensors=True).to(device)
    segformer_model.eval()
    
    for filename in tqdm(os.listdir(input_folder), desc="Creating Ground Masks"):
        base_name, _ = os.path.splitext(filename)
        input_path = os.path.join(input_folder, filename)
        #output_path = f"GroundMasks/{base_name}_ground.png"
        
        #print(f"basename: {base_name}")
        #print(f"filename: {filename}")
        
        #output_path = f"GroundMasks/{input_folder}/{base_name}_ground.png"
        #output_path_folder = f"GroundMasks/{input_folder}"
        
        #output_path = f"{ground_dir}{base_name}_ground.png"
        #output_path_folder = f"{ground_dir}"
        
        output_path = os.path.join(ground_dir, f"{base_name}_ground.png")
        output_path_folder = ground_dir

        if not os.path.exists(output_path_folder):
            os.makedirs(output_path_folder)

        if not os.path.exists(output_path):
            create_ground_mask(input_path, output_path, feature_extractor, segformer_model, device)

    print("creating ground masks is done", flush=True)


def extract_masks(semantic_map):
    """Extract binary masks for road and sidewalk etc. from a semantic segmentation map."""

    road_color = np.array([128, 64, 128])
    sidewalk_color = np.array([232, 35, 244])
    parking_color = np.array([160, 170, 250]) 
    ground_color = np.array([81, 0, 81])

    road_mask = np.all(semantic_map == road_color, axis=-1).astype(np.uint8) * 255
    sidewalk_mask = np.all(semantic_map == sidewalk_color, axis=-1).astype(np.uint8) * 255
    parking_mask = np.all(semantic_map == parking_color, axis=-1).astype(np.uint8) * 255
    ground_mask = np.all(semantic_map == ground_color, axis=-1).astype(np.uint8) * 255

    return road_mask, sidewalk_mask, parking_mask, ground_mask



def shrink_mask(mask, kernel_size=5, iterations=1):

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    shrunk_mask = cv2.erode(mask, kernel, iterations=iterations)
    
    return shrunk_mask