import argparse
import os
import sys

# Append parent dir to path if running directly so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
# Import the baseline model globally constructed in utils.load_model
from utils.load_model import model, processor
from utils.generation import generate_answers_for_pope, generate_answers_for_chair

def main():
    parser = argparse.ArgumentParser(description="Baseline Generation for POPE and CHAIR using LLaVA-1.5")
    parser.add_argument("--run_pope", action="store_true", help="Run POPE generation")
    parser.add_argument("--run_chair", action="store_true", help="Run CHAIR generation")
    parser.add_argument("--pope_output", default="results/baseline_pope.json", type=str)
    parser.add_argument("--chair_output", default="results/baseline_chair.json", type=str)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--coco_dir", type=str, default=None, help="Local directory containing COCO val2014 images")
    parser.add_argument("--coco_annotations", type=str, default=None, help="Path to COCO instances_val2014.json")
    
    args = parser.parse_args()
    
    if not (args.run_pope or args.run_chair):
        print("Please specify --run_pope and/or --run_chair")
        return

    # POPE Routine
    if args.run_pope:
        print("Loading POPE dataset...")
        try:
            pope_dataset = load_dataset("lmms-lab/POPE", split="test")
            
            if args.limit is not None:
                print(f"Limiting POPE to {args.limit} samples.")
                pope_dataset = pope_dataset.select(range(min(args.limit, len(pope_dataset))))
                
            generate_answers_for_pope(
                model=model,
                processor=processor,
                dataset=pope_dataset,
                output_file=args.pope_output
            )
            print(f"==> POPE generation complete. Saved to {args.pope_output}")
        except Exception as e:
            print(f"Failed to run POPE: {e}")

    # CHAIR Routine
    if args.run_chair:
        print("Preparing dataset for CHAIR...")
        chair_dataset = []
        
        # Method 1: Load from local COCO directory combined with annotations
        if args.coco_dir and args.coco_annotations:
            import json
            from PIL import Image
            
            print(f"Loading local COCO from {args.coco_dir}...")
            with open(args.coco_annotations, 'r') as f:
                coco_gt = json.load(f)
                
            for img_info in coco_gt['images']:
                img_path = os.path.join(args.coco_dir, img_info['file_name'])
                if os.path.exists(img_path):
                    chair_dataset.append({
                        "image_id": img_info['id'],
                        "image": Image.open(img_path).convert("RGB")
                    })
                    if args.limit and len(chair_dataset) >= args.limit:
                        break
        else:
            # Method 2: Fallback to try loading a default HF COCO validation dataset
            print("No local COCO provided. Attempting to load HuggingFaceM4/COCO...")
            try:
                coco_hf = load_dataset("HuggingFaceM4/COCO", split="validation")
                if args.limit is not None:
                    coco_hf = coco_hf.select(range(min(args.limit, len(coco_hf))))
                    
                for item in coco_hf:
                    # Depending on exact structure of HF COCO
                    image_id = item.get('image_id', item.get('cocoid', None))
                    if image_id is None:
                        continue
                        
                    chair_dataset.append({
                        "image_id": image_id,
                        "image": item['image'].convert("RGB")
                    })
            except Exception as e:
                print(f"Failed to load HuggingFace dataset automatically: {e}")
                print("For CHAIR, please provide --coco_dir and --coco_annotations for your local MS-COCO val2014.")
                
        if chair_dataset:
            generate_answers_for_chair(
                model=model,
                processor=processor,
                images_dataset=chair_dataset,
                output_file=args.chair_output
            )
            print(f"==> CHAIR generation complete. Saved to {args.chair_output}")

if __name__ == "__main__":
    main()
