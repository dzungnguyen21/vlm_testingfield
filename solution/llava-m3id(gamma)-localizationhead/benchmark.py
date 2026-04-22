import argparse
import os
import sys
import json
from tqdm import tqdm

# Append parent dir to path if running directly so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from model import load_m3id_plus_engine

def generate_answers_for_pope(engine, dataset, output_file):
    predictions = []
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Starting M3ID+ generation for POPE ({len(dataset)} examples)...")
    for item in tqdm(dataset):
        image = item['image'] # PIL.Image
        question = item['question']
        q_id = item.get('question_id', item.get('id', None))
        
        # M3ID+ Engine generate() method automatically prepends USER: <image>...\nASSISTANT:
        try:
            output_text = engine.generate(prompt=question, image_path=image, max_new_tokens=10, visualize=False)
        except Exception as e:
            print(f"Error generating for image {q_id}: {e}")
            output_text = ""
            
        predictions.append({
            "question_id": q_id,
            "text": output_text
        })
        
    print(f"Saving predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
        
    return predictions

def generate_answers_for_chair(engine, images_dataset, output_file, prompt="Please describe this image in detail."):
    predictions = []
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    print(f"Starting M3ID+ CHAIR caption generation ({len(images_dataset)} examples)...")
    for item in tqdm(images_dataset):
        image = item['image'] # PIL.Image
        image_id = item['image_id']
        
        try:
            output_text = engine.generate(prompt=prompt, image_path=image, max_new_tokens=100, visualize=False)
        except Exception as e:
            print(f"Error generating for image {image_id}: {e}")
            output_text = ""
            
        predictions.append({
            "image_id": image_id,
            "caption": output_text
        })
        
    print(f"Saving CHAIR predictions to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
        
    return predictions

def main():
    parser = argparse.ArgumentParser(description="M3ID+ Gamma Generation for POPE and CHAIR")
    parser.add_argument("--run_pope", action="store_true", help="Run POPE generation")
    parser.add_argument("--run_chair", action="store_true", help="Run CHAIR generation")
    parser.add_argument("--pope_output", default="results/m3id_gamma_pope.json", type=str)
    parser.add_argument("--chair_output", default="results/m3id_gamma_chair.json", type=str)
    parser.add_argument("--weights_path", default="gamma_net_weights_full.pth", type=str, help="Path to the gamma_net_weights_full.pth file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--coco_dir", type=str, default=None, help="Local directory containing COCO val2014 images")
    parser.add_argument("--coco_annotations", type=str, default=None, help="Path to COCO instances_val2014.json")
    
    args = parser.parse_args()
    
    if not (args.run_pope or args.run_chair):
        print("Please specify --run_pope and/or --run_chair")
        return

    # Initialize M3ID+ Engine
    # This will throw FileNotFoundError if weights_path is not uploaded
    print(f"Loading M3ID+ Engine with weights: {args.weights_path}")
    engine = load_m3id_plus_engine(weights_path=args.weights_path)

    # POPE Routine
    if args.run_pope:
        print("Loading POPE dataset...")
        try:
            pope_dataset = load_dataset("lmms-lab/POPE", split="test")
            
            if args.limit is not None:
                print(f"Limiting POPE to {args.limit} samples.")
                pope_dataset = pope_dataset.select(range(min(args.limit, len(pope_dataset))))
                
            generate_answers_for_pope(
                engine=engine,
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
        
        # Method 1: Load from local COCO directory directly without parsing massive JSON
        if args.coco_dir:
            from PIL import Image
            
            print(f"Loading local COCO from {args.coco_dir}...")
            
            if os.path.exists(args.coco_dir):
                for filename in os.listdir(args.coco_dir):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        img_path = os.path.join(args.coco_dir, filename)
                        
                        try:
                            # Safely extract digits from COCO format: COCO_val2014_000000391895.jpg
                            image_id_str = ''.join(filter(str.isdigit, filename.split('_')[-1]))
                            if not image_id_str:
                                continue
                            image_id = int(image_id_str)
                        except Exception:
                            continue
                            
                        chair_dataset.append({
                            "image_id": image_id,
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
                engine=engine,
                images_dataset=chair_dataset,
                output_file=args.chair_output
            )
            print(f"==> CHAIR generation complete. Saved to {args.chair_output}")
        else:
            print(f"Warning: No valid CHAIR images found in {args.coco_dir} matching {args.coco_annotations}")

if __name__ == "__main__":
    main()
