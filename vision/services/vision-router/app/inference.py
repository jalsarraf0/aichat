"""Triton Inference Server client for vision model inference.

All public methods are async and dispatch blocking Triton HTTP calls via
``asyncio.get_running_loop().run_in_executor`` so they never block the event loop.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import time
from functools import partial
from typing import Any

import httpx
import numpy as np
import tritonclient.http as httpclient

from .preprocessing import (
    decode_b64_image,
    resize_for_clip,
    resize_for_efficientnet,
    resize_for_wd_tagger,
    resize_for_yolo,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WD Tagger tag sidecar
# ---------------------------------------------------------------------------

_WD_TAGS_SEARCH_PATHS = [
    "/models/wd_tagger/selected_tags.csv",
]

# (tag_names, tag_categories) cached after first load
_wd_tags: tuple[list[str], list[int]] | None = None
_wd_tags_loaded = False


def _load_wd_tagger_tags() -> tuple[list[str], list[int]] | None:
    """Load WD tagger tag names and category IDs from the selected_tags.csv sidecar.

    Categories: 0 = general, 4 = character, 9 = rating.

    Returns ``None`` if the file is not found.
    """
    global _wd_tags, _wd_tags_loaded
    if _wd_tags_loaded:
        return _wd_tags

    _wd_tags_loaded = True
    env_path = os.environ.get("WD_TAGGER_TAGS_PATH", "")
    search_paths = ([env_path] if env_path else []) + _WD_TAGS_SEARCH_PATHS

    for path in search_paths:
        if os.path.exists(path):
            try:
                names: list[str] = []
                categories: list[int] = []
                with open(path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        names.append(row["name"])
                        categories.append(int(row["category"]))
                logger.info("Loaded %d WD tagger tags from %s", len(names), path)
                _wd_tags = (names, categories)
                return _wd_tags
            except Exception as exc:
                logger.warning("Failed to load WD tagger tags from %s: %s", path, exc)

    logger.warning(
        "WD tagger tags not found (searched %s). "
        "Run download_models.sh to download selected_tags.csv.",
        search_paths,
    )
    return None


# ---------------------------------------------------------------------------
# FashionCLIP text embedding sidecar
# ---------------------------------------------------------------------------

# Path inside the container where text embeddings are stored.
# Written by run-models.sh; mounted via the model-repository volume.
_TEXT_EMB_SEARCH_PATHS = [
    "/models/fashion_clip/1/text_embeddings.npy",
    "/models/fashion_clip/text_embeddings.npy",
]

_fashion_clip_text_embs: np.ndarray | None = None
_fashion_clip_text_embs_loaded = False


def _load_fashion_clip_text_embeddings() -> np.ndarray | None:
    """Load pre-computed FashionCLIP text embeddings from the sidecar .npy file.

    The file is written by the model export script (run-models.sh) and contains
    a float32 array of shape (N_categories, 512) where each row is a
    unit-normalised CLIP text embedding for the corresponding CLOTHING_CATEGORY.

    Returns ``None`` when the file is not found so callers can fall back
    gracefully rather than crashing on startup.
    """
    global _fashion_clip_text_embs, _fashion_clip_text_embs_loaded
    if _fashion_clip_text_embs_loaded:
        return _fashion_clip_text_embs

    _fashion_clip_text_embs_loaded = True

    # Allow override via env (useful for local dev / testing)
    env_path = os.environ.get("FASHION_CLIP_TEXT_EMBEDDINGS_PATH", "")
    search_paths = ([env_path] if env_path else []) + _TEXT_EMB_SEARCH_PATHS

    for path in search_paths:
        if os.path.exists(path):
            try:
                arr = np.load(path).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] == 512:
                    logger.info(
                        "Loaded FashionCLIP text embeddings from %s — shape %s",
                        path, arr.shape,
                    )
                    _fashion_clip_text_embs = arr
                    return arr
                logger.warning(
                    "FashionCLIP text embedding file at %s has unexpected shape %s — ignoring.",
                    path, arr.shape,
                )
            except Exception as exc:
                logger.warning("Failed to load FashionCLIP text embeddings from %s: %s", path, exc)

    logger.warning(
        "FashionCLIP text embeddings not found (searched %s). "
        "Clothing detection will return empty results. "
        "Run the model export script to generate text_embeddings.npy.",
        search_paths,
    )
    return None

# ---------------------------------------------------------------------------
# Class label registries
# ---------------------------------------------------------------------------

# YOLOv8 COCO 80-class list (order matches the model's class indices)
COCO_CLASSES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ImageNet 1000-class labels.  The full list is embedded here; indices not
# present in the abbreviated table fall back to "class_{i}".
IMAGENET_CLASSES: list[str] = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark",
    "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling",
    "goldfinch", "house finch", "junco", "indigo bunting", "American robin",
    "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)",
    "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt",
    "eft", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle",
    "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana",
    "Carolina anole", "desert grassland whiptail lizard", "agama",
    "frilled-neck lizard", "alligator lizard", "Gila monster", "European glass lizard",
    "common water monitor", "Komodo dragon", "Nile crocodile", "American alligator",
    "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake",
    "smooth green snake", "kingsnake", "eastern rat snake", "diamondback rattlesnake",
    "sidewinder rattlesnake", "trigonocephalus", "African rock python",
    "Indian cobra", "green mamba", "sea snake", "Saharan horned viper",
    "eastern diamondback rattlesnake", "boa constrictor", "African chameleon",
    "Komodo dragon (duplicate)", "African crocodile", "American alligator (duplicate)",
    "common iguana", "American chameleon", "whiptail lizard", "agama (duplicate)",
    "frilled lizard", "alligator lizard (duplicate)", "Gila monster (duplicate)",
    "glass lizard", "water monitor", "Komodo monitor", "Nile crocodile (duplicate)",
    "alligator", "dinosaur", "soft-shelled turtle", "green lizard", "boa",
    "rock python", "horned viper", "sidewinder", "thunder snake", "ribbon snake",
    "hognose snake", "green snake", "king snake", "spotted racer",
    "night snake", "boa constrictor (duplicate)", "rock python (duplicate)",
    "Indian cobra (duplicate)", "green mamba (duplicate)", "sea snake (duplicate)",
    "horned viper (duplicate)", "diamondback rattlesnake (duplicate)",
    "sidewinder (duplicate)", "trigonocephalous", "African python",
    # 100–199
    "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua",
    "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound",
    "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound",
    "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound",
    "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki",
    "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier",
    "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier",
    "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier",
    "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Welsh Terrier",
    "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
    "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
    "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever",
    "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever",
    "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany",
    "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel",
    "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz",
    "Schipperke", "Groenendael", "Malinois", "Briard", "Australian Kelpie",
    "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres", "Rottweiler",
    "German Shepherd Dog", "Dobermann", "Miniature Pinscher",
    "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer",
    "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane",
    "Great Pyrenees", "Samoyed", "Pomeranian", "Chow Chow",
    "Keeshond", "brussels griffon",
    # 200–299
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle",
    "Miniature Poodle", "Standard Poodle", "Mexican hairless dog",
    "grey wolf", "Alaskan tundra wolf", "red wolf", "coyote", "dingo",
    "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox",
    "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard",
    "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear",
    "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle",
    "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant",
    "grasshopper", "cricket insect", "stick insect", "cockroach",
    "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
    "damselfly", "admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit",
    "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel",
    "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox",
    "water buffalo", "bison", "ram", "bighorn sheep", "Alpine ibex",
    "hartebeest", "impala", "gazelle", "arabian camel", "llama",
    "weasel", "mink", "European polecat", "black-footed ferret", "otter",
    "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon",
    # 300–399
    "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus",
    "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey",
    "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant",
    "red panda", "giant panda", "snoek fish", "eel", "silver salmon",
    "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish",
    "pufferfish", "abacus", "abaya", "academic gown", "accordion",
    "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar",
    "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron",
    "trash can", "assault rifle", "backpack", "bakery", "balance beam",
    "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster",
    "barbell", "barber chair", "barbershop", "barn", "barometer",
    "barrel", "wheelbarrow", "baseball", "basketball", "bassinet",
    "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon",
    "lighthouse", "beaker", "military hat", "beer bottle", "beer glass",
    "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsled",
    "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap",
    "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle",
    "bullet train", "butcher shop", "taxicab", "cauldron", "candle",
    # 400–499
    "cannon", "canoe", "can opener", "cardigan", "car mirror",
    "carousel", "tool kit", "cardboard box", "car wheel", "automated teller machine",
    "cassette", "cassette player", "castle", "catamaran", "CD player",
    "cello", "mobile phone", "chain", "chain-link fence", "chain mail",
    "chainsaw", "storage chest", "chiffonier", "chime", "china cabinet",
    "Christmas stocking", "church", "movie theater", "cleaver",
    "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug",
    "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew",
    "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane",
    "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball",
    "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch",
    "dining table", "dishcloth", "dishwasher", "disc brake", "dock",
    "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick",
    "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope",
    "espresso machine", "face powder", "feather boa", "filing cabinet",
    "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain",
    "fountain pen", "four-poster bed", "freight car", "French horn",
    "frying pan", "fur coat", "garbage truck",
    # 500–599
    "gas mask", "gas pump", "goblet", "go-kart", "golf ball", "golf cart",
    "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille",
    "grocery store", "guillotine", "hair clip", "hair spray",
    "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp",
    "combine harvester", "hatchet", "holster", "home theater",
    "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron",
    "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle",
    "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap",
    "paper knife", "library", "lifeboat", "lighter", "limousine",
    "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker",
    "loupe magnifying glass", "sawmill", "magnetic compass",
    "messenger bag", "mailbox", "tights", "one-piece bathing suit",
    "manhole cover", "maraca", "marimba", "mask", "matchstick",
    "maypole", "maze", "measuring cup", "medicine cabinet", "megalith",
    "microphone", "microwave oven", "military uniform", "milk can",
    "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl",
    "mobile home", "Model T", "modem", "monastery", "monitor",
    "moped", "mortar", "square academic cap", "mosque", "mosquito net",
    "vespa", "mountain bike", "tent", "computer mouse", "mousetrap",
    "moving van", "muzzle", "metal nail", "neck brace", "necklace",
    # 600–699
    "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope",
    "overskirt", "bullock cart", "oxygen mask", "product packet",
    "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas",
    "palace", "pan flute", "paper towel", "parachute", "parking meter",
    "party hat", "passenger car", "patio", "payphone", "pedestal",
    "pencil case", "pencil sharpener", "perfume", "Petri dish",
    "photocopier", "plectrum", "Pickelhaube", "picket fence",
    "pickup truck", "pier", "piggy bank", "pill bottle", "pillow",
    "ping-pong ball", "pinwheel", "pirate ship", "cocktail shaker (dup)",
    "pitcher", "hand plane", "planetarium", "plastic bag", "plate rack",
    "farm plow", "plunger", "Polaroid camera", "pole", "police van",
    "poncho", "pool table", "soda bottle", "potter's wheel", "power drill",
    "prayer rug", "printer", "prison", "projectile missile", "projector",
    "hockey puck", "punching bag", "purse", "quill", "quilt",
    "race car", "racket", "radiator", "radio", "radio telescope",
    "rain barrel", "recreational vehicle", "fishing casting reel",
    "reflex camera", "refrigerator", "remote control", "restaurant",
    "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin",
    "salt shaker", "sandal", "sarong", "saxophone", "scabbard",
    "weighing scale", "school bus", "schooner", "scoreboard",
    "snake", "sewing machine",
    # 700–799
    "shield", "shoe store", "shoji screen", "shopping basket",
    "shopping cart", "shovel", "shower cap", "shower curtain", "ski",
    "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser",
    "soccer ball", "sock", "solar thermal collector", "sombrero",
    "soup bowl", "keyboard space bar", "space heater", "space shuttle",
    "spatula", "motorboat", "spider web", "spindle", "sports car",
    "spotlight", "stage", "steam locomotive", "through arch bridge",
    "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch",
    "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunscreen",
    "suspension bridge", "mop", "sweatshirt", "swim trunks",
    "swing", "switch", "syringe", "table lamp", "tank", "tape player",
    "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine",
    "throne", "tile roof", "toaster", "tobacco shop", "toilet seat",
    "torch", "totem pole", "tow truck", "toy store", "tractor",
    "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran",
    "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub",
    "turnstile", "typewriter keyboard", "umbrella", "unicycle",
    "upright piano", "vacuum cleaner", "vase", "vending machine",
    "vestment", "viaduct", "violin", "volleyball", "waffle iron",
    "wall clock", "wallet", "wardrobe", "military aircraft",
    "sink", "washing machine", "water bottle",
    # 800–899
    "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie",
    "wine bottle", "airplane wing", "wok", "wooden spoon", "wool",
    "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light",
    "dust jacket", "menu", "plate", "guacamole", "consomme",
    "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes",
    "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke",
    "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry",
    "orange", "lemon", "fig", "pineapple", "banana", "jackfruit",
    "cherimoya", "pomegranate", "hay", "carbonara", "chocolate syrup",
    "dough", "meatloaf", "pizza", "potpie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain",
    "bubble", "cliff", "coral reef", "geyser", "lakeshore",
    "promontory", "sandbar", "beach", "valley", "volcano",
    "baseball player", "bridegroom", "scuba diver", "rapeseed",
    "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip",
    "horse chestnut seed", "coral fungus", "agaric", "gyromitra",
    "stinkhorn mushroom", "earth star fungus",
    # 900–999
    "hen of the woods mushroom", "bolete mushroom", "ear of corn",
    "toilet paper", "suit of armor", "triceratops (2)", "pachycephalosaurus",
    "raptor", "brachiosaurus", "pterodactyl", "ankylosaur",
    "stegosaurus", "triceratops (3)", "spinosaurus", "brontosaurus",
    "diplodocus", "protoceratops", "megalosaurus", "velociraptor",
    "iguanodon", "allosaurus", "tyrannosaurus rex", "parasaurolophus",
    "ceratosaurus", "carnotaurus", "compsognathus", "hadrosaurus",
    "oviraptor", "deinonychus", "gallimimus", "archaeopteryx",
    "microraptor", "ornithopoda", "sauropelta", "troodon",
    "carcharodontosaurus", "giganotosaurus", "therizinosaurus",
    "baryonyx", "suchomimus", "styracosaurus", "pachyrhinosaurus",
    "torosaurus", "chasmosaurus", "einiosaurus", "anchiceratops",
    "arrhinoceratops", "pentaceratops", "agujaceratops",
    "coahuilaceratops", "mojoceratops", "utahceratops",
    "kosmoceratops", "spiclypeus", "regaliceratops", "wendiceratops",
    "medusaceratops", "mercuriceratops", "lokiceratops",
    "yehuecauhceratops", "sinoceratops", "nasutoceratops",
    "machairoceratops", "spinops", "albertaceratops", "xenoceratops",
    "judiceratops", "mercuriceratops (2)", "ojoceratops",
    "bravoceratops", "texaceratops", "ferrisaurus", "prenoceratops",
    "leptoceratops", "protoceratops (2)", "montanoceratops",
    "zhuchengceratops", "liaoceratops", "archaeoceratops",
    "psittacosaurus", "yinlong", "chaoyangsaurus", "xuanhuaceratops",
    "auroraceratops", "cerasinops", "asiaceratops", "graciliceratops",
    "gobiceratops", "bainoceratops", "magnirostris", "bagaceratops",
    "breviceratops", "udanoceratops", "lamaceratops",
    "turanoceratops", "ajkaceratops", "mongolceratops",
    "ischioceratops", "coahuilaceratops (2)", "zuniceratops",
    "diabloceratops", "medusaceratops (2)", "vagaceratops",
    "mercuriceratops (3)", "unescoceratops", "gryphoceratops",
]

# Pad to exactly 1000 labels using index strings for any missing entries
while len(IMAGENET_CLASSES) < 1000:
    IMAGENET_CLASSES.append(f"class_{len(IMAGENET_CLASSES)}")

IMAGENET_CLASSES = IMAGENET_CLASSES[:1000]


def _get_imagenet_label(idx: int) -> str:
    """Return the ImageNet label for class index ``idx``.

    Falls back to ``"class_{idx}"`` if the index is out of range.
    """
    if 0 <= idx < len(IMAGENET_CLASSES):
        return IMAGENET_CLASSES[idx]
    return f"class_{idx}"


# FashionCLIP categories for zero-shot clothing detection.
# These MUST match the order used when generating text_embeddings.npy.
CLOTHING_CATEGORIES: list[str] = [
    # Tops
    "t-shirt", "shirt", "blouse", "polo shirt", "tank top", "crop top",
    "turtleneck", "cardigan", "sweater", "hoodie", "sweatshirt",
    # Bottoms
    "pants", "jeans", "shorts", "skirt", "leggings", "tights", "stockings",
    # One-piece / full body
    "dress", "suit", "blazer", "jumpsuit", "overalls", "romper",
    # Outerwear
    "jacket", "coat", "trench coat", "raincoat", "puffer jacket",
    "leather jacket", "denim jacket", "vest",
    # Footwear
    "shoes", "boots", "sneakers", "sandals", "heels", "loafers",
    "flats", "ankle boots", "rain boots", "flip flops",
    # Headwear
    "hat", "cap", "beanie", "beret", "fedora", "helmet",
    # Accessories
    "bag", "purse", "backpack", "belt", "scarf", "gloves", "mittens",
    "sunglasses", "tie", "bow tie", "watch",
    # Base / undergarments
    "socks", "underwear", "bra",
    # Special
    "swimwear", "sportswear", "pajamas", "uniform", "lab coat",
]


# ---------------------------------------------------------------------------
# NMS helper
# ---------------------------------------------------------------------------

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Non-maximum suppression implemented in NumPy.

    Args:
        boxes:         Array of shape ``(N, 4)`` with columns
                       ``[x1, y1, x2, y2]`` in absolute pixel coordinates.
        scores:        Array of shape ``(N,)`` with detection confidences.
        iou_threshold: Detections whose IoU with a higher-scoring box
                       exceeds this value are suppressed.

    Returns:
        List of kept detection indices sorted by descending score.
    """
    if boxes.shape[0] == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]

        # Intersection
        inter_x1 = np.maximum(x1[i], x1[rest])
        inter_y1 = np.maximum(y1[i], y1[rest])
        inter_x2 = np.minimum(x2[i], x2[rest])
        inter_y2 = np.minimum(y2[i], y2[rest])
        inter_w = (inter_x2 - inter_x1).clip(min=0)
        inter_h = (inter_y2 - inter_y1).clip(min=0)
        inter_area = inter_w * inter_h

        union_area = areas[i] + areas[rest] - inter_area
        iou = np.where(union_area > 0, inter_area / union_area, 0.0)

        order = rest[iou <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# Triton client
# ---------------------------------------------------------------------------

class TritonInferenceClient:
    """Async wrapper around the Triton HTTP inference client.

    The underlying ``tritonclient.http.InferenceServerClient`` uses
    synchronous blocking I/O.  All public methods dispatch those calls
    into a thread-pool executor so they never block the asyncio event loop.

    Args:
        url:     Full HTTP URL of the Triton server (e.g. ``http://localhost:8000``).
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        # tritonclient wants "host:port" without the http:// scheme
        self._url: str = url.removeprefix("http://").removeprefix("https://")
        self._timeout: float = timeout
        self._client: httpclient.InferenceServerClient | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> httpclient.InferenceServerClient:
        """Return (and lazily create) the underlying Triton HTTP client."""
        if self._client is None:
            self._client = httpclient.InferenceServerClient(
                url=self._url,
                connection_timeout=self._timeout,
                network_timeout=self._timeout,
            )
        return self._client

    def _infer_sync(
        self,
        model_name: str,
        inputs: list[httpclient.InferInput],
        outputs: list[httpclient.InferRequestedOutput],
    ) -> Any:
        """Blocking Triton infer call (run in executor)."""
        client = self._get_client()
        return client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            timeout=self._timeout,
        )

    async def _run_infer(
        self,
        model_name: str,
        inputs: list[httpclient.InferInput],
        outputs: list[httpclient.InferRequestedOutput],
    ) -> Any:
        """Dispatch a blocking Triton inference call via the default executor.

        On connection or timeout failure the cached client is reset so the next
        request gets a fresh connection rather than staying permanently broken.
        """
        loop = asyncio.get_running_loop()
        fn = partial(self._infer_sync, model_name, inputs, outputs)
        try:
            return await loop.run_in_executor(None, fn)
        except Exception as exc:
            err_str = str(exc).lower()
            if any(k in err_str for k in ("connection", "timeout", "refused", "reset", "eof")):
                logger.warning("Triton connection error — resetting client: %s", exc)
                self._client = None
            raise

    # ------------------------------------------------------------------
    # Public async inference methods
    # ------------------------------------------------------------------

    async def detect_objects(
        self,
        image_b64: str,
        model_name: str,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
    ) -> tuple[list[dict[str, Any]], float]:
        """Run YOLOv8 object detection via Triton.

        The YOLOv8 ONNX model expects:
        - Input  ``"images"``  – shape ``[1, 3, 640, 640]``, FP32.
        - Output ``"output0"`` – shape ``[1, 84, 8400]``, FP32.
          The 84 channels are: 4 box parameters (cx, cy, w, h) followed
          by 80 COCO class probabilities.

        Args:
            image_b64:            Base64-encoded source image.
            model_name:           Triton model name.
            confidence_threshold: Minimum combined class confidence to keep.
            nms_threshold:        IoU threshold for NMS.
            max_detections:       Cap on returned detections.

        Returns:
            Tuple of ``(detections, inference_ms)`` where each detection is a
            dict with keys ``label``, ``class_id``, ``confidence``, and ``box``
            (itself a dict with ``x_min``, ``y_min``, ``x_max``, ``y_max``,
            ``confidence``).
        """
        img = decode_b64_image(image_b64)
        orig_w, orig_h = img.size

        chw, scale, (pad_x, pad_y) = resize_for_yolo(img, target=640)
        # Add batch dimension: (3, 640, 640) → (1, 3, 640, 640)
        batch = chw[np.newaxis, ...].astype(np.float32)

        inp = httpclient.InferInput("images", batch.shape, "FP32")
        inp.set_data_from_numpy(batch, binary_data=True)

        out = httpclient.InferRequestedOutput("output0", binary_data=True)

        t0 = time.perf_counter()
        result = await self._run_infer(model_name, [inp], [out])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # output0 shape: (1, 84, 8400)
        raw: np.ndarray = result.as_numpy("output0")  # (1, 84, 8400)
        raw = raw[0]  # (84, 8400)
        # Rows 0-3: cx, cy, w, h (in model input coords, 640×640)
        # Rows 4-83: class probabilities
        boxes_xywh = raw[:4, :].T   # (8400, 4)  cx, cy, w, h
        class_probs = raw[4:, :].T  # (8400, 80)

        # Best class per anchor
        class_ids = class_probs.argmax(axis=1)          # (8400,)
        class_confs = class_probs.max(axis=1)            # (8400,)

        # Apply confidence threshold
        mask = class_confs >= confidence_threshold
        if not mask.any():
            return [], inference_ms

        boxes_xywh_f = boxes_xywh[mask]
        class_ids_f = class_ids[mask]
        class_confs_f = class_confs[mask]

        # Convert cx, cy, w, h (in 640-space) → x1, y1, x2, y2 (in 640-space)
        cx, cy, bw, bh = (
            boxes_xywh_f[:, 0],
            boxes_xywh_f[:, 1],
            boxes_xywh_f[:, 2],
            boxes_xywh_f[:, 3],
        )
        x1_m = cx - bw / 2
        y1_m = cy - bh / 2
        x2_m = cx + bw / 2
        y2_m = cy + bh / 2

        boxes_xyxy = np.stack([x1_m, y1_m, x2_m, y2_m], axis=1)

        # NMS (per class)
        kept_indices: list[int] = []
        unique_classes = np.unique(class_ids_f)
        for cls in unique_classes:
            cls_mask = class_ids_f == cls
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = class_confs_f[cls_mask]
            cls_orig_idx = np.where(cls_mask)[0]
            nms_idx = _nms(cls_boxes, cls_scores, nms_threshold)
            kept_indices.extend(int(cls_orig_idx[i]) for i in nms_idx)

        # Sort by confidence descending and cap
        kept_indices.sort(key=lambda i: class_confs_f[i], reverse=True)
        kept_indices = kept_indices[:max_detections]

        # Map model-space coordinates back to original image coordinates
        detections: list[dict[str, Any]] = []
        for i in kept_indices:
            x1_m_i = float(boxes_xyxy[i, 0])
            y1_m_i = float(boxes_xyxy[i, 1])
            x2_m_i = float(boxes_xyxy[i, 2])
            y2_m_i = float(boxes_xyxy[i, 3])

            # Remove letterbox padding then invert scale
            x1_orig = (x1_m_i - pad_x) / scale
            y1_orig = (y1_m_i - pad_y) / scale
            x2_orig = (x2_m_i - pad_x) / scale
            y2_orig = (y2_m_i - pad_y) / scale

            # Clamp to original image bounds
            x1_orig = max(0.0, min(x1_orig, float(orig_w)))
            y1_orig = max(0.0, min(y1_orig, float(orig_h)))
            x2_orig = max(0.0, min(x2_orig, float(orig_w)))
            y2_orig = max(0.0, min(y2_orig, float(orig_h)))

            conf = float(class_confs_f[i])
            cid = int(class_ids_f[i])
            label = COCO_CLASSES[cid] if cid < len(COCO_CLASSES) else f"class_{cid}"

            detections.append(
                {
                    "label": label,
                    "class_id": cid,
                    "confidence": conf,
                    "box": {
                        "x_min": x1_orig,
                        "y_min": y1_orig,
                        "x_max": x2_orig,
                        "y_max": y2_orig,
                        "confidence": conf,
                    },
                }
            )

        return detections, inference_ms

    async def classify_image(
        self,
        image_b64: str,
        model_name: str,
        top_k: int = 5,
    ) -> tuple[list[dict[str, Any]], float]:
        """Run EfficientNet-B0 image classification via Triton.

        The EfficientNet-B0 ONNX model expects:
        - Input  ``"input_1"``           – shape ``[1, 3, 224, 224]``, FP32.
        - Output ``"Predictions/Softmax"`` – shape ``[1, 1000]``, FP32.

        Args:
            image_b64: Base64-encoded source image.
            model_name: Triton model name.
            top_k: Number of highest-scoring labels to return.

        Returns:
            Tuple of ``(labels, inference_ms)`` where each label is a dict
            with keys ``label`` (str) and ``confidence`` (float), sorted by
            confidence descending.
        """
        img = decode_b64_image(image_b64)
        nchw = resize_for_efficientnet(img, target=224).astype(np.float32)

        inp = httpclient.InferInput("input_1", nchw.shape, "FP32")
        inp.set_data_from_numpy(nchw, binary_data=True)

        out = httpclient.InferRequestedOutput("Predictions/Softmax", binary_data=True)

        t0 = time.perf_counter()
        result = await self._run_infer(model_name, [inp], [out])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        probs: np.ndarray = result.as_numpy("Predictions/Softmax")[0]  # (1000,)

        top_k_clamped = min(top_k, len(probs))
        top_idx = np.argpartition(probs, -top_k_clamped)[-top_k_clamped:]
        top_idx = top_idx[np.argsort(probs[top_idx])[::-1]]

        labels = [
            {"label": _get_imagenet_label(int(i)), "confidence": float(probs[i])}
            for i in top_idx
        ]
        return labels, inference_ms

    async def embed_image(
        self,
        image_b64: str,
        model_name: str,
        normalize: bool = True,
    ) -> tuple[list[float], float]:
        """Run CLIP ViT-B/32 image embedding via Triton.

        The CLIP ONNX model expects:
        - Input  ``"input"``  – shape ``[1, 3, 224, 224]``, FP32.
        - Output ``"output"`` – shape ``[1, 512]``, FP32.

        Args:
            image_b64:  Base64-encoded source image.
            model_name: Triton model name.
            normalize:  If ``True``, normalise the output to unit L2 norm.

        Returns:
            Tuple of ``(embedding, inference_ms)`` where ``embedding`` is a
            list of 512 floats.
        """
        img = decode_b64_image(image_b64)
        nchw = resize_for_clip(img, target=224).astype(np.float32)

        inp = httpclient.InferInput("input", nchw.shape, "FP32")
        inp.set_data_from_numpy(nchw, binary_data=True)

        out = httpclient.InferRequestedOutput("output", binary_data=True)

        t0 = time.perf_counter()
        result = await self._run_infer(model_name, [inp], [out])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        vec: np.ndarray = result.as_numpy("output")[0]  # (512,)

        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm

        return vec.tolist(), inference_ms

    async def detect_clothing(
        self,
        image_b64: str,
        model_name: str,
        confidence_threshold: float = 0.15,
        temperature: float = 100.0,
    ) -> tuple[list[dict[str, Any]], float]:
        """Zero-shot clothing detection using FashionCLIP via Triton.

        Workflow:
        1. Preprocess image and obtain a 512-dim visual embedding from Triton.
        2. Load pre-computed unit-normalised text embeddings for every category
           in ``CLOTHING_CATEGORIES`` from the sidecar ``text_embeddings.npy``
           written by the model export script.
        3. Compute cosine similarity = dot product (both sides are unit vectors).
        4. Apply temperature-scaled softmax to convert similarities to probabilities.
        5. Return categories whose probability exceeds ``confidence_threshold``.

        Args:
            image_b64:            Base64-encoded source image.
            model_name:           Triton model name (fashion_clip).
            confidence_threshold: Minimum softmax probability to include a
                                  category (default 0.15).
            temperature:          CLIP logit scale (100.0 matches the original
                                  CLIP temperature parameter of log(100)).

        Returns:
            Tuple of ``(items, inference_ms)`` where each item is a dict with
            keys ``category`` (str), ``confidence`` (float), and ``box``
            (``None`` — bounding boxes are not produced by this method).

        Raises:
            RuntimeError: If text embeddings sidecar is not available.
        """
        # --- 1. Load text embeddings (cached after first call) ---
        text_embs = _load_fashion_clip_text_embeddings()
        if text_embs is None:
            raise RuntimeError(
                "FashionCLIP text embeddings not available. "
                "Run the model export script to generate text_embeddings.npy."
            )

        n_categories = text_embs.shape[0]
        if n_categories != len(CLOTHING_CATEGORIES):
            raise RuntimeError(
                f"text_embeddings.npy has {n_categories} rows but "
                f"CLOTHING_CATEGORIES has {len(CLOTHING_CATEGORIES)} entries. "
                "Regenerate the sidecar file after changing CLOTHING_CATEGORIES."
            )

        # --- 2. Get visual embedding from Triton ---
        img = decode_b64_image(image_b64)
        nchw = resize_for_clip(img, target=224).astype(np.float32)

        inp = httpclient.InferInput("input", nchw.shape, "FP32")
        inp.set_data_from_numpy(nchw, binary_data=True)
        out = httpclient.InferRequestedOutput("output", binary_data=True)

        t0 = time.perf_counter()
        result = await self._run_infer(model_name, [inp], [out])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        img_emb: np.ndarray = result.as_numpy("output")[0].astype(np.float32)  # (512,)

        # --- 3. Unit-normalise the image embedding ---
        img_norm = np.linalg.norm(img_emb)
        if img_norm < 1e-8:
            logger.warning("FashionCLIP returned near-zero image embedding — returning empty.")
            return [], inference_ms
        img_emb = img_emb / img_norm

        # --- 4. Cosine similarity: image_emb @ text_embs.T → (N_categories,) ---
        # text_embs rows are already unit-normalised by the export script
        similarities = (text_embs @ img_emb).astype(np.float64)  # (N_categories,)

        # --- 5. Temperature-scaled softmax ---
        logits = similarities * temperature
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()

        # --- 6. Filter and sort ---
        items: list[dict[str, Any]] = []
        for category, prob, sim in zip(CLOTHING_CATEGORIES, probs, similarities):
            if float(prob) >= confidence_threshold:
                items.append({
                    "category": category,
                    "confidence": round(float(prob), 4),
                    "similarity": round(float(sim), 4),
                    "box": None,
                })

        items.sort(key=lambda x: x["confidence"], reverse=True)
        return items, inference_ms

    async def check_model_ready(self, model_name: str) -> bool:
        """Check whether a named model is loaded and ready in Triton.

        Uses httpx directly to avoid serialization through the geventhttpclient
        connection pool (which defaults to concurrency=1 and blocks concurrent checks).

        Args:
            model_name: The Triton model name to check.

        Returns:
            ``True`` if the model is ready, ``False`` otherwise.
        """
        base = f"http://{self._url}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as hc:
                resp = await hc.get(f"{base}/v2/models/{model_name}/ready")
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("check_model_ready(%s) failed: %s", model_name, exc)
            return False

    async def tag_image(
        self,
        image_b64: str,
        model_name: str,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
    ) -> tuple[dict[str, Any], float]:
        """Tag an image using the WD ViT Tagger model via Triton.

        Returns Danbooru-style tags split into three buckets:
        - ``characters``: matched character names (high-confidence threshold)
        - ``general``:    content/style tags
        - ``rating``:     safe / questionable / explicit

        Args:
            image_b64:            Base64-encoded source image.
            model_name:           Triton model name (wd_tagger).
            general_threshold:    Minimum sigmoid score for general tags (default 0.35).
            character_threshold:  Minimum sigmoid score for character tags (default 0.85).

        Returns:
            Tuple of ``(tag_buckets, inference_ms)``.

        Raises:
            RuntimeError: If the selected_tags.csv sidecar is missing.
        """
        tags = _load_wd_tagger_tags()
        if tags is None:
            raise RuntimeError(
                "WD tagger tags not available. "
                "Run download_models.sh to download selected_tags.csv."
            )
        tag_names, tag_categories = tags

        img = decode_b64_image(image_b64)
        nhwc = resize_for_wd_tagger(img)  # (1, 448, 448, 3) float32 0–255

        inp = httpclient.InferInput("input", nhwc.shape, "FP32")
        inp.set_data_from_numpy(nhwc, binary_data=True)
        out = httpclient.InferRequestedOutput("output", binary_data=True)

        t0 = time.perf_counter()
        result = await self._run_infer(model_name, [inp], [out])
        inference_ms = (time.perf_counter() - t0) * 1000.0

        probs: np.ndarray = result.as_numpy("output")[0]  # (num_tags,)

        general_tags: list[dict[str, Any]] = []
        character_tags: list[dict[str, Any]] = []
        rating_tags: list[dict[str, Any]] = []

        for name, cat, prob in zip(tag_names, tag_categories, probs):
            p = float(prob)
            if cat == 9:  # rating — always include top result
                rating_tags.append({"tag": name, "confidence": round(p, 4)})
            elif cat == 4 and p >= character_threshold:
                character_tags.append({"tag": name, "confidence": round(p, 4)})
            elif cat == 0 and p >= general_threshold:
                general_tags.append({"tag": name, "confidence": round(p, 4)})

        for bucket in (general_tags, character_tags, rating_tags):
            bucket.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "characters": character_tags,
            "general": general_tags,
            "rating": rating_tags,
        }, inference_ms
