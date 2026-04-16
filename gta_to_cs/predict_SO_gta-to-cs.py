from torch.utils.data import dataset
import deeplab as deeplab
import utils
import os
import random
import argparse
import numpy as np
 
from torch.utils import data
from dataload.cityscapes_predict import  Cityscapes

from torchvision import transforms as T
import torchvision.transforms.functional as TF

import torch
import torch.nn as nn

from PIL import Image
from utils import ext_transforms as et


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes',"geo"], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in deeplab.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              deeplab.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    #parser.add_argument("--input_images", type=str, required=True,
                        #help="path to a single image or image directory")
    
    parser.add_argument("--predicted_images", type=str, required=True,
                        help="path to a single image or image directory")
    
    return parser

class TransformDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            # Assuming data is a tuple of (image, label)
            image, label, name= data
            image ,label= self.transform(image,label)
            return image, label,name
        #return data
    
    def __len__(self):
        return len(self.dataset)
def main():
    opts = get_argparser().parse_args()

    if opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target



    #os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    test_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    city_test= Cityscapes(root="datasets/cityscapes/",
                                    split='val')
    city_test = TransformDataset(city_test, transform=test_transform)

    
    target_testLoader = data.DataLoader(
            city_test, batch_size=8, shuffle=False, num_workers=2,pin_memory=True,drop_last=False)

 

    
    # Set up model (all models are 'constructed at network.modeling)
    model = deeplab.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        deeplab.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt,map_location=torch.device('cpu'))
        model = nn.DataParallel(model)
        model.module.load_state_dict(checkpoint["model_state"],strict=False)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images
    model.eval()



    #os.makedirs(opts.input_images, exist_ok=True)
    os.makedirs(opts.predicted_images, exist_ok=True)


    with torch.no_grad():
        img_count = 0
        for i, (images, labels, img_names) in enumerate(target_testLoader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs= model(images)
            preds = outputs.detach().max(dim=1)[1]

            decode_fn = Cityscapes.decode_target

            for j in range(images.size(0)):
                pred = preds[j].cpu().numpy()
                decoded_pred = decode_fn(pred).astype(np.uint8)

                pred_path = os.path.join(opts.predicted_images, f"{img_names[j]}_pred.png")

                Image.fromarray(decoded_pred).save(pred_path)





if __name__ == '__main__':
    main()