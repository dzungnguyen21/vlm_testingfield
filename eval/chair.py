'''
Adapted from the official CHAIR metric provided by the user.
Modified to maintain compatibility with the Colab notebook command line arguments:
--input_file, --output_file, and gracefully handle missing Train annotations.
'''

import os
import sys
import nltk
import json
import argparse
import tqdm
import pickle
from collections import defaultdict

# Try to download NLTK data. In production, this should be handled once during setup.
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK datasets for CHAIR metric...")
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nltk.download('punkt')

# Try pattern, fallback to nltk lemmatizer
try:
    from pattern.en import singularize
except ImportError:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    def singularize(word):
        return _lemmatizer.lemmatize(word)

synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''

def combine_coco_captions(annotation_path):
    val_path = '%s/captions_val2014.json' % annotation_path
    train_path = '%s/captions_train2014.json' % annotation_path
    
    if not os.path.exists(val_path):
        raise Exception(f"Please download MSCOCO caption annotations for val set. Missing: {val_path}")

    val_caps = json.load(open(val_path))
    
    images = val_caps['images']
    annotations = val_caps['annotations']
    
    # Gracefully handle missing train sets
    if os.path.exists(train_path):
        train_caps = json.load(open(train_path))
        images += train_caps['images']
        annotations += train_caps['annotations']

    all_caps = {
        'info': val_caps.get('info', {}),
        'licenses': val_caps.get('licenses', []),
        'images': images,
        'annotations': annotations
    }
    return all_caps 

def combine_coco_instances(annotation_path):
    val_path = '%s/instances_val2014.json' % annotation_path
    train_path = '%s/instances_train2014.json' % annotation_path
    
    if not os.path.exists(val_path):
        raise Exception(f"Please download MSCOCO instance annotations for val set. Missing: {val_path}")

    val_instances = json.load(open(val_path))
    
    images = val_instances['images']
    annotations = val_instances['annotations']

    if os.path.exists(train_path):
        train_instances = json.load(open(train_path))
        images += train_instances['images']
        annotations += train_instances['annotations']

    all_instances = {
        'info': val_instances.get('info', {}),
        'licenses': val_instances.get('licenses', []),
        'categories': val_instances['categories'],
        'images': images,
        'annotations': annotations
    }
    return all_instances 

class CHAIR(object):
    def __init__(self, coco_path):
        self.imid_to_objects = defaultdict(list)
        self.coco_path = coco_path

        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms if s.strip()]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light', 'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case', 'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie', 'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal', 'cub']
        vehicle_words = ['jet', 'train']
        
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' %animal_word] = animal_word
            self.double_word_dict['adult %s' %animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' %vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'
        
        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        self.caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.caps) == len(self.eval_imids)

    def caption_to_words(self, caption):
        words = nltk.word_tokenize(caption.lower())
        words = [singularize(w) for w in words]
    
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
           idxs.append(i) 
           double_word = ' '.join(words[i:i+2])
           if double_word in self.double_word_dict: 
               double_words.append(self.double_word_dict[double_word])
               i += 2
           else:
               double_words.append(words[i])
               i += 1
        words = double_words
    
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']
    
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        id_to_name = {}
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks" %(i, len(segment_annotations)))
            imid = annotation['image_id']
            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        # We only use captions if they exist
        try:
            coco_caps = combine_coco_captions(self.coco_path)
            caption_annotations = coco_caps['annotations']

            for i, annotation in enumerate(caption_annotations):
                sys.stdout.write('\rGetting annotations for %d/%d ground truth captions' %(i, len(caption_annotations)))
                imid = annotation['image_id']
                _, node_words, _, _ = self.caption_to_words(annotation['caption'])
                self.imid_to_objects[imid].extend(node_words)
            print("\n")
        except Exception as e:
            print(f"Skipping ground truth caption annotation loading: {e}")

    def get_annotations(self):
        self.get_annotations_from_segments() 
        self.get_annotations_from_captions()
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, cap_file, image_id_key, caption_key):
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)
        
        imid_to_objects = self.imid_to_objects
        caps = self.caps
        eval_imids = self.eval_imids
 
        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        
        num_recall_gt_objects = 0.
        num_gt_objects = 0.

        output = {'sentences': []} 
        
        for i in tqdm.trange(len(caps)):
            cap :str = caps[i]
            imid :int = eval_imids[i]
    
            words, node_words, idxs, raw_words = self.caption_to_words(cap) 
 
            # If ground truth objects are entirely missing for this image, initialize empty
            gt_objects = imid_to_objects.get(imid, set())
            
            cap_dict = {'image_id': imid, 
                        'caption': cap,
                        'mscoco_hallucinated_words': [],
                        'mscoco_gt_words': list(gt_objects),
                        'mscoco_generated_words': list(node_words),
                        'hallucination_idxs': [], 
                        'words': raw_words 
                        }

            cap_dict['metrics'] = {'CHAIRs': 0, 'CHAIRi': 0, 'Recall': 0}
 
            coco_word_count += len(node_words) 
            hallucinated = False
            
            recall_gt_objects = set()
            for word, node_word, idx in zip(words, node_words, idxs):
                if node_word not in gt_objects:
                    hallucinated_word_count += 1 
                    cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                    cap_dict['hallucination_idxs'].append(idx)
                    hallucinated = True
                else:
                    recall_gt_objects.add(node_word)
    
            num_caps += 1
            if hallucinated:
               num_hallucinated_caps += 1
            
            num_gt_objects += len(gt_objects)
            num_recall_gt_objects += len(recall_gt_objects)
    
            cap_dict['metrics']['CHAIRs'] = int(hallucinated)
            cap_dict['metrics']['CHAIRi'] = 0.
            cap_dict['metrics']['Recall'] = 0.
            
            if len(words) > 0:
                cap_dict['metrics']['CHAIRi'] = len(cap_dict['mscoco_hallucinated_words'])/float(len(words))
            
            if len(gt_objects) > 0:
                cap_dict['metrics']['Recall'] = len(recall_gt_objects) / len(gt_objects)
   
            output['sentences'].append(cap_dict)
 
        chair_s = (num_hallucinated_caps/num_caps) if num_caps > 0 else 0.
        chair_i = (hallucinated_word_count/coco_word_count) if coco_word_count > 0 else 0.
        recall = (num_recall_gt_objects / num_gt_objects) if num_gt_objects > 0 else 0.
    
        output['overall_metrics'] = {'CHAIRs': chair_s, 'CHAIRi': chair_i, 'Recall': recall}
        return output 

def load_generated_captions(cap_file, image_id_key:str, caption_key:str):
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        caps = json.load(open(cap_file))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f'Unsupported extension {ext} for cap_file: {cap_file}')

    imids = [int(obj[image_id_key]) for obj in caps]
    caps = [obj[caption_key] for obj in caps]
       
    return caps, imids

def save_hallucinated_words(cap_file, cap_dict): 
    os.makedirs(os.path.dirname(os.path.abspath(cap_file)), exist_ok=True)
    with open(cap_file, 'w') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)

def print_metrics(hallucination_cap_dict, quiet=False):
    sentence_metrics = hallucination_cap_dict['overall_metrics']
    
    for k, v in sentence_metrics.items():
        k_str = str(k).ljust(10)
        v_str = f'{v * 100:.01f}'
        print(k_str, v_str, sep=': ')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # New official script args
    parser.add_argument("--cap_file", type=str, default='',
                        help="path towards json or jsonl saving image ids and their captions in list of dict.")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="in each dict of cap_file, which key stores image id of coco.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="in each dict of cap_file, which key stores caption of the image.")
    parser.add_argument("--cache", type=str, default="chair.pkl",
                        help="pre inited CHAIR evaluator object, for fast loading.")
    parser.add_argument("--coco_path", type=str, default='',
                        help="only use for regenerating CHAIR evaluator object, will be ignored if uses cached evaluator.")
    parser.add_argument("--save_path", type=str, default="",
                        help="saving CHAIR evaluate and results to json, useful for debugging the caption model.")
                        
    # Maintain backwards compatibility with earlier Colab framework Notebook args
    parser.add_argument("--input_file", type=str, default="", help="Alias for --cap_file")
    parser.add_argument("--output_file", type=str, default="", help="Alias for --save_path")
    parser.add_argument("--coco_annotations", type=str, default="", help="Path to instances_val2014.json")
    
    args = parser.parse_args()

    # Apply aliases
    cap_file_path = args.input_file if args.input_file else args.cap_file
    save_file_path = args.output_file if args.output_file else args.save_path
    
    # If coco_path is not defined but coco_annotations is (notebook legacy)
    c_path = args.coco_path
    if not c_path and args.coco_annotations:
        c_path = os.path.dirname(args.coco_annotations)
        if not c_path: c_path = '.'

    if args.cache and os.path.exists(args.cache):
        evaluator = pickle.load(open(args.cache, 'rb'))
        print(f"Loaded evaluator from cache: {args.cache}")
    else:
        print(f"Cache not set or not exist yet, building from scratch using path: {c_path}...")
        evaluator = CHAIR(c_path)
        if args.cache:
            try:
                pickle.dump(evaluator, open(args.cache, 'wb'))
                print(f"Cached evaluator to: {args.cache}")
            except Exception as e:
                print(f"Could not write cache {args.cache}: {e}")

    cap_dict = evaluator.compute_chair(cap_file_path, args.image_id_key, args.caption_key) 
    
    print_metrics(cap_dict)
    
    if save_file_path:
        save_hallucinated_words(save_file_path, cap_dict)
        print(f"Saved extended output to {save_file_path}")
