import os
import argparse
from depth_map import batch_create_depth_maps
from ground_mask import batch_create_ground_masks
from weather_render import batch_render_weather

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    parser = argparse.ArgumentParser(description="Weather Rendering CLI")
    
    # directories 
    parser.add_argument("--input_dir", type=str, default="Input", help="Path to input image folder")
    parser.add_argument("--output_dir", type=str, default="Output", help="Path to output image folder")
    parser.add_argument("--depth_dir", type=str, default="DepthMaps", help = "Path to depthmask folder")
    parser.add_argument("--ground_dir", type=str, default="GroundMasks", help = "Path to groundmask folder")
    
    
    # rain intensity and outputs
    parser.add_argument("--intensity", type=float, default=None, help="Rain intensity (0 to 100)")
    parser.add_argument("--save_base", action="store_true", help="Save base rain+fog version")
    parser.add_argument("--save_wet", action="store_true", help="Save rain+fog+wet version")
    parser.add_argument("--save_puddles", action="store_true", help="Save rain+fog+wet+puddles version")

    
    
    ### optional parameters for additional parameter sensitivity experiments. 
    parser.add_argument("--resolution", type=int, default=None, help="Override Perlin noise resolution")
    parser.add_argument("--persistence", type=float, default=None, help="Override fractal noise persistence")
    parser.add_argument("--reflection", type=float, default=None, help="Override puddle reflection coefficient")
    parser.add_argument("--beta", type=float, default=None, help="Override atmospheric haze beta")

    args = parser.parse_args()

    if not (args.save_base or args.save_wet or args.save_puddles):
        args.save_base = True
        args.save_wet = True
        args.save_puddles = True

    rain_intensity = args.intensity
    if rain_intensity is not None:
        rain_intensity = max(0, min(100, rain_intensity))

    ensure_dir(args.output_dir)
    if args.save_base:
        ensure_dir(os.path.join(args.output_dir, "Base"))
    if args.save_wet:
        ensure_dir(os.path.join(args.output_dir, "Wet"))
    if args.save_puddles:
        ensure_dir(os.path.join(args.output_dir, "Puddles"))
    ensure_dir(args.depth_dir)
    ensure_dir(args.ground_dir)

    # STEP 1: Create all depth maps
    batch_create_depth_maps(args.input_dir, args.depth_dir)

    # STEP 2: Create all ground masks
    batch_create_ground_masks(args.input_dir, args.ground_dir)
    
    print('done creating masks', flush=True)

    # STEP 3: Render weather
    batch_render_weather(
            intensity_level=rain_intensity,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            depth_dir=args.depth_dir,
            ground_dir=args.ground_dir,
            save_base=args.save_base,
            save_wet=args.save_wet,
            save_puddles=args.save_puddles,
            beta=args.beta,
            res_value=args.res,
            persistence_value=args.persistence,
            reflection_value=args.reflection,


        )

if __name__ == "__main__":
    main()