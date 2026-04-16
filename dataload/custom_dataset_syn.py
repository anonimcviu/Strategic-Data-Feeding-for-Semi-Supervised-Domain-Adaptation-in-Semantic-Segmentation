
import os
from torch.utils.data import Dataset 
from PIL import Image
import numpy as np
import cv2


class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder): #, transform=None):
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

        # Read label using the same method as Synthia reference
        label_cv = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        label = label_cv[..., 2]  # keep only last channel
        label = label.astype(np.int32)
        target = encode_labels(label)  # your function
        target_pil = Image.fromarray(target)
        return img, target_pil


        # Apply transformations
        #if self.transform:
        #    image, target = self.transform(img,label)
        
        #print("LABELS unique2", np.unique(target_pil))

 
    
mapping_20 = {
   0:255, # void x
   1:9, # sky
   2: 2, # building
   3: 0, # road
   4: 1, # sidewalk
   5: 4, # fence
   6: 8, # vegetation
   7: 5, # pole
   8: 12, # car
   9: 7, # traffic sign
   10: 10, # pedestrian/person
   11: 15, # bicycle
   12:  14, # motorcycle
   13: 255, # Parking-slot x
   14: 255, # Road-work x
   15:  6,  # traffic light
   16: 255, # terrain x
   17: 11 , # rider
   18: 255, # truck x
   19: 13, #bus
   20: 255, # train x
   21: 3, # wall
   22: 255, #lanemarking x
}

def encode_labels(mask):
    label_mask = np.full_like(mask, 255)  # start with all void (255)
    for src_id, dst_id in mapping_20.items():
        label_mask[mask == src_id] = dst_id
    return label_mask

    

    
    
