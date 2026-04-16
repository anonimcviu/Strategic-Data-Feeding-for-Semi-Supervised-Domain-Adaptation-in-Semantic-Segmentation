
import os
from torch.utils.data import Dataset 
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder):   #, transform=None
        self.image_folder = image_folder
        self.label_folder = label_folder
        #self.transform = transform

        # Get list of image and label files
        self.image_files = sorted(os.listdir(image_folder))
        self.label_files = sorted(os.listdir(label_folder))

        # Ensure the number of images and labels match
        assert len(self.image_files) == len(self.label_files), "Number of images and labels must be the same"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])

        img = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # Apply transformations
        #if self.transform:
        #    image, target = self.transform(img,label)
        
        #print("shape1",label.size)
        label = np.array(label).astype(np.int32) 
        target=encode_labels(label)
        target_pil = Image.fromarray(target.astype(np.uint8))
        #print("shape2",target_pil.size)
        #print("label unique", np.unique(target))

        return img, target_pil
    

    
    
mapping_20 = {
        0: 255,
        1: 255,
        2: 255,
        3: 255,
        4: 255,
        5: 255,
        6: 255,
        7: 0,
        8: 1,
        9: 255,
        10: 255,
        11: 2,
        12: 3,
        13: 4,
        14: 255,
        15: 255,
        16: 255,
        17: 5,
        18: 255,
        19: 6,
        20: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        29: 255,
        30: 255,
        31: 16,
        32: 17,
        33: 18,
        34:255
    }

def encode_labels(mask):
    label_mask = np.zeros_like(mask)
    for k in mapping_20:
        label_mask[mask == k] = mapping_20[k]
    return label_mask
    

    
    
