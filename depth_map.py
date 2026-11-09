import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import contextlib
import logging
import sys
import os
from tqdm import tqdm

@contextlib.contextmanager
def suppress_all_output():
    
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull

        logging_disabled = logging.root.manager.disable
        logging.disable(logging.CRITICAL)

        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logging.disable(logging_disabled)
            
            

def create_depth_map(img_path, depth_map_path, model, device):

    def resize_for_midas(image):
        new_width = int(np.ceil(image.size[0] / 32)) * 32
        new_height = int(np.ceil(image.size[1] / 32)) * 32
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    transform = Compose([
        Resize(384), 
        resize_for_midas,  
        ToTensor(), 
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img_pil = Image.fromarray(img)

    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)

    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    output_path = depth_map_path
    cv2.imwrite(output_path, depth_map_normalized)




def batch_create_depth_maps(input_folder, depth_dir):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Midas Using device: {device}")

    
    # Load MiDaS Small from PyTorch Hub
    with suppress_all_output():
        depth_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small"
        )

    # Alternative: load from a local MiDaS repository
    # repo_path = "/path/to/MiDaS"
    # depth_model = torch.hub.load(
    #     repo_path, "MiDaS_small", source="local"
    # )
    
    
    depth_model.to(device)
    depth_model.eval()

    os.makedirs(depth_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder), desc="Creating Depth Maps"):
        base_name, _ = os.path.splitext(filename)
        input_path = os.path.join(input_folder, filename)
        
        
        output_path = os.path.join(depth_dir, f"{base_name}_depth.png")
        

        if not os.path.exists(output_path):
            create_depth_map(input_path, output_path, model=depth_model, device=device)
