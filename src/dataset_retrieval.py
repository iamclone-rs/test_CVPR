import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

unseen_classes = {
    'Sketchy': [
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
    ],
    'tuberlin': [
        "banana",
        "bus",
        "tractor",
        "suitcase",
        "streetlight",
        "telephone",
        "bottle opener",
        "canoe",
        "fan",
        "teacup",
        "penguin",
        "laptop",
        "shoe",
        "lighter",
        "hot air balloon",
        "pizza",
        "brain",
        "ant",
        "t-shirt",
        "trombone",
        "windmill",
        "snowboard",
        "table",
        "rollerblades",
        "parachute",
        "space shuttle",
        "bridge",
        "frying-pan",
        "bread",
        "horse",
    ]
}

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False, image_type='triplet'):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig
        self.image_type = image_type

        # Support folder structure for both Sketchy and Tuberlin if consistent
        # For Tuberlin, if structure is different, adjustments might be needed here
        # Assuming standard structure: data_dir/sketch/category/*.png and data_dir/photo/category/*.jpg
        if not os.path.exists(os.path.join(self.opts.data_dir, 'sketch')):
             # Fallback or check if data_dir directly contains categories? 
             # For now assume structure is maintained.
             pass

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        current_unseen = unseen_classes.get(self.opts.dataset, unseen_classes['Sketchy'])

        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(current_unseen))
            else:
                self.all_categories = current_unseen

        self.all_sketches_path = []
        self.all_photos_path = {}
        self.all_photos_flat = []

        def get_files(path, extensions):
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(path, ext)))
            return files

        sk_exts = ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.PNG', '*.JPG', '*.JPEG', '*.SVG']
        im_exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']

        valid_categories = []
        for category in self.all_categories:
            sk_files = get_files(os.path.join(self.opts.data_dir, 'sketch', category), sk_exts)
            ph_files = get_files(os.path.join(self.opts.data_dir, 'photo', category), im_exts)
            
            if len(sk_files) > 0 and len(ph_files) > 0:
                self.all_sketches_path.extend(sk_files)
                self.all_photos_path[category] = ph_files
                for ph in ph_files:
                    self.all_photos_flat.append((ph, category))
                valid_categories.append(category)
            else:
                print(f"Warning: Category '{category}' dropped. Sketches: {len(sk_files)}, Photos: {len(ph_files)}")
        
        self.all_categories = valid_categories

    def __len__(self):
        if self.image_type == 'gallery':
            return len(self.all_photos_flat)
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        if self.image_type == 'gallery':
            filepath, category = self.all_photos_flat[index]
            filename = os.path.basename(filepath)
            img_data = ImageOps.pad(Image.open(filepath).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
            img_tensor = self.transform(img_data)
            # Returning dummy values for sketch and neg to keep signature consistent if needed, 
            # OR better: return just what is needed and handle in collate/model.
            # But the model expects unpacked values.
            # Model validation_step expects: sk_tensor, img_tensor, neg_tensor, category = batch[:4]
            # If we change return signature, we must handle it in validation_step.
            return (torch.zeros_like(img_tensor), img_tensor, torch.zeros_like(img_tensor), category, filename)

        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
