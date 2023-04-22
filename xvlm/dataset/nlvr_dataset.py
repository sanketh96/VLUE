import json
import os
from torch.utils.data import Dataset
from PIL import Image
from dataset.utils import pre_caption
import random


class nlvr_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = 30
        if len(self.ann) > 10000:
            self.subsample_n = 100
            self.ann = random.sample(self.ann, self.subsample_n)
            print("Size after sampling : ", len(self.ann))
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image0_path = os.path.join(self.image_root, ann['images'][0])
        image0 = Image.open(image0_path).convert('RGB')
        image0 = self.transform(image0)

        image1_path = os.path.join(self.image_root, ann['images'][1])
        image1 = Image.open(image1_path).convert('RGB')
        image1 = self.transform(image1)

        sentence = pre_caption(ann['sentence'], self.max_words)

        if ann['label'] == 'True':
            label = 1

        elif ann['label'] == 'False':
            label = 0

        else:
            raise ValueError(f"unsupported label: {ann['label']}")

        return image0, image1, sentence, label
