import os
import glob
import re
import random
from collections import defaultdict

import torch
from PIL import Image, ImageOps
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


SKETCHY_UNSEEN_CLASSES = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]


def default_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]),
    ])


def _list_files(directory, extensions):
    files = set()
    for extension in extensions:
        files.update(glob.glob(os.path.join(directory, extension)))
    return sorted(files)


def _normalize_stem(stem):
    return stem.strip().lower()


def _candidate_photo_keys(sketch_stem):
    sketch_stem = _normalize_stem(sketch_stem)
    candidates = [sketch_stem]

    # Sketchy FG sketch files typically follow:
    # photo_stem-1.png, photo_stem-2.png, ...
    match = re.match(r"^(.*)-(\d+)$", sketch_stem)
    if match:
        candidates.append(match.group(1))

    # Fall back to a single underscore suffix strip only when the
    # resulting key exists as a standalone photo stem in the dataset.
    match = re.match(r"^(.*)_(\d+)$", sketch_stem)
    if match:
        candidates.append(match.group(1))
    return list(dict.fromkeys(candidates))


def _match_photo_key(sketch_stem, photo_keys):
    candidates = _candidate_photo_keys(sketch_stem)
    for candidate in candidates:
        if candidate in photo_keys:
            return candidate

    normalized_stem = _normalize_stem(sketch_stem)
    for photo_key in sorted(photo_keys, key=len, reverse=True):
        if normalized_stem == photo_key:
            return photo_key
        if any(normalized_stem.startswith(photo_key + sep) for sep in ('-', '_', ' ')):
            return photo_key
    return None


class SketchyFGDataset(torch.utils.data.Dataset):
    def __init__(self, opts, transform=None, split='train', view='train'):
        super().__init__()
        self.opts = opts
        self.transform = transform or default_transform(opts.max_size)
        self.split = split
        self.view = view

        self.sketch_root = os.path.join(self.opts.data_dir, 'sketch')
        self.photo_root = os.path.join(self.opts.data_dir, 'photo')
        self.sketch_extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
        self.photo_extensions = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')

        categories = sorted([
            category for category in os.listdir(self.sketch_root)
            if os.path.isdir(os.path.join(self.sketch_root, category))
        ])
        if split == 'train':
            self.categories = [category for category in categories if category not in SKETCHY_UNSEEN_CLASSES]
        else:
            self.categories = [category for category in categories if category in SKETCHY_UNSEEN_CLASSES]

        self.records = []
        self.gallery_records = []
        self.photos_by_category_instance = defaultdict(lambda: defaultdict(list))

        self._build_records()
        if self.view == 'train':
            self._build_train_index()

    def _build_records(self):
        dropped_sketches = 0

        for category in self.categories:
            sketch_dir = os.path.join(self.sketch_root, category)
            photo_dir = os.path.join(self.photo_root, category)
            if not os.path.isdir(photo_dir):
                continue

            photo_paths = _list_files(photo_dir, self.photo_extensions)
            photo_key_to_paths = defaultdict(list)
            for photo_path in photo_paths:
                photo_stem = _normalize_stem(os.path.splitext(os.path.basename(photo_path))[0])
                photo_key_to_paths[photo_stem].append(photo_path)
                self.gallery_records.append({
                    'category': category,
                    'instance_id': photo_stem,
                    'photo_path': photo_path,
                })
                self.photos_by_category_instance[category][photo_stem].append(photo_path)

            sketch_paths = _list_files(sketch_dir, self.sketch_extensions)
            for sketch_path in sketch_paths:
                sketch_stem = os.path.splitext(os.path.basename(sketch_path))[0]
                photo_key = _match_photo_key(sketch_stem, photo_key_to_paths.keys())
                if photo_key is None:
                    dropped_sketches += 1
                    continue

                positive_photo_path = photo_key_to_paths[photo_key][0]
                self.records.append({
                    'category': category,
                    'instance_id': photo_key,
                    'sketch_path': sketch_path,
                    'photo_path': positive_photo_path,
                })

        if self.view == 'gallery':
            self.records = self.gallery_records

        if self.view == 'train':
            filtered_records = []
            for record in self.records:
                category_instances = self.photos_by_category_instance[record['category']]
                if len(category_instances) > 1:
                    filtered_records.append(record)
            self.records = filtered_records

        print(
            f"[{self.split}/{self.view}] categories={len(self.categories)} "
            f"records={len(self.records)} dropped_sketches={dropped_sketches}"
        )

    def _build_train_index(self):
        self.instance_keys_by_category = {}
        for category, instances in self.photos_by_category_instance.items():
            keys = sorted(instances.keys())
            if len(keys) > 1:
                self.instance_keys_by_category[category] = keys

    def __len__(self):
        return len(self.records)

    def _load_image(self, path):
        image = Image.open(path).convert('RGB')
        image = ImageOps.pad(image, size=(self.opts.max_size, self.opts.max_size))
        return self.transform(image)

    def __getitem__(self, index):
        record = self.records[index]
        category = record['category']
        instance_id = record['instance_id']

        if self.view == 'gallery':
            photo_tensor = self._load_image(record['photo_path'])
            return photo_tensor, category, instance_id

        sketch_tensor = self._load_image(record['sketch_path'])

        if self.view == 'query':
            return sketch_tensor, category, instance_id

        photo_tensor = self._load_image(record['photo_path'])
        negative_instance_choices = [
            key for key in self.instance_keys_by_category[category]
            if key != instance_id
        ]
        negative_instance_id = random.choice(negative_instance_choices)
        negative_photo_path = random.choice(self.photos_by_category_instance[category][negative_instance_id])
        negative_tensor = self._load_image(negative_photo_path)
        return sketch_tensor, photo_tensor, negative_tensor, category, instance_id
