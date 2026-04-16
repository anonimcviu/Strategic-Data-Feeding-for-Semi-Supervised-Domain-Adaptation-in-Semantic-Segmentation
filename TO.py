from tqdm import tqdm
import deeplab as deeplab
import utils
#import os
import random
import argparse
import numpy as np
from torch.utils import data
from dataload import Cityscapes
from utils import ext_transforms as et
#from metrics import StreamSegMetrics
from torch.utils.data import random_split
import torch
import torch.nn as nn
from dataload import CustomDataset
#from sklearn.model_selection import train_test_split
import wandb
from torchmetrics import JaccardIndex , Accuracy
from torch.cuda.amp import autocast, GradScaler
torch.cuda.empty_cache()

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes','gta',], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--num_epoch", type=int, default=16)

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
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=17091997,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    

    parser.add_argument("--height", type=int, help="Desired height of the output image.")
    parser.add_argument("--width", type=int, help="Desired width of the output image.")

    parser.add_argument("--crop_size_h", type=int, help="Desired height of the output image.")
    parser.add_argument("--crop_size_w", type=int, help="Desired width of the output image.")


    parser.add_argument("--run_name", type=str, help="run name")

    parser.add_argument("--deterministic", action='store_true', default=False)

    parser.add_argument("--itrs_goal", type=int, default=80e3,
                        help="epoch number (default: 30k)")


    return parser


class TransformDataset:
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            # Assuming data is a tuple of (image, label)
            image, label = data
            image ,label= self.transform(image,label)
            return image, label
        #return data
    
    def __len__(self):
        return len(self.dataset)


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = et.ExtCompose([
            et.ExtResize(height=1052,width=1914),
            et.ExtRandomCrop(size=(opts.crop_size_h, opts.crop_size_w)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    val_transform = et.ExtCompose([
            et.ExtResize( height=1052,width=1914),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    train_transform_cs = et.ExtCompose([
            et.ExtRandomCrop(size=(opts.crop_size_h, opts.crop_size_w)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    val_transform_cs = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    test_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    
    
    #GTA
    train_set_gta= CustomDataset(image_folder= opts.data_root + "/gta/train_full/images/", label_folder=opts.data_root+ "/gta/train_full/labels/")
    val_set_gta= CustomDataset(image_folder= opts.data_root + "/gta/test/images/", label_folder=opts.data_root+ "/gta/test/labels/")

    gta_train = TransformDataset(train_set_gta, transform=train_transform)
    gta_val = TransformDataset(val_set_gta, transform=val_transform)

    #CS
    city_train= Cityscapes(root=opts.data_root + "/cityscapes/",split='train')
    train_size = int(0.9 * len(city_train))
    val_size = len(city_train) - train_size
    cs_train, cs_val = random_split(city_train, [train_size, val_size], generator=torch.Generator().manual_seed(opts.random_seed))

    #transform
    cs_train = TransformDataset(cs_train, transform=train_transform_cs)
    cs_val = TransformDataset(cs_val, transform=val_transform_cs)

    if opts.dataset == 'gta':
        return gta_train,gta_val,cs_val
    
    if opts.dataset == 'cityscapes':

        if opts.test_only:
            city_test= Cityscapes(root=opts.data_root + "/cityscapes/", split='val')
            city_test = TransformDataset(city_test, transform=test_transform)                             
            return city_test       

        return cs_train,cs_val,gta_val 


def validate(opts, model, loader, device):
    """Do validation and return specified samples"""
    #metrics.reset()

    accuracy_overall = Accuracy(task="multiclass", num_classes=opts.num_classes,average="micro",ignore_index=255).to(device)
    jaccard_overall = JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="micro",ignore_index=255).to(device)
    accuracy_mean = Accuracy(task="multiclass", num_classes=opts.num_classes,average="macro",ignore_index=255).to(device)
    jaccard_mean= JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="macro",ignore_index=255).to(device)
    accuracy_cls = Accuracy(task="multiclass", num_classes=opts.num_classes,average="none",ignore_index=255).to(device)
    jaccard_cls= JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="none",ignore_index=255).to(device)

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1]
            #targets = labels

            # Update metrics in batch
            jaccard_overall.update(preds, labels)
            accuracy_overall.update(preds, labels)
            jaccard_mean.update(preds, labels)
            accuracy_mean.update(preds, labels)
            jaccard_cls.update(preds, labels)
            accuracy_cls.update(preds, labels)

        # Compute metrics
        score = {
            "Overall IOU": float(jaccard_overall.compute().item()),
            "Overall ACC": float(accuracy_overall.compute().item()),
            "Mean IOU": float(jaccard_mean.compute().item()),
            "Mean ACC": float(accuracy_mean.compute().item()),
            "Class-wise IOU": jaccard_cls.compute(),
            "Class-wise ACC": accuracy_cls.compute(),
        }
        score_copy=score.copy()

        # Print results
        for metric, value in score.items():
            print(f"{metric}: {value}")

        # Reset metrics for future use
        for metric in [jaccard_overall, accuracy_overall, jaccard_mean, accuracy_mean, jaccard_cls, accuracy_cls]:
            metric.reset()

    return score_copy

def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

def main():
    opts = get_argparser().parse_args()
    
    opts.num_classes = 19

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    torch.cuda.manual_seed_all(opts.random_seed)

    if opts.deterministic:
        torch.backends.cudnn.deterministic = True

    # Setup dataloader
    

    if opts.test_only:
        test_dst= get_dataset(opts)
        target_testLoader = data.DataLoader(
        test_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=1,pin_memory=True,drop_last=True)
        print("Dataset: %s,  Test set: %d" %
            (opts.dataset, len(test_dst)))
    else:
        train_dst, val_dst, val_dst_t = get_dataset(opts)
        source_trainLoader = data.DataLoader(
            train_dst, batch_size=opts.batch_size, shuffle=True, 
            num_workers=2,pin_memory=True, drop_last=True,worker_init_fn=worker_init_fn)  # drop_last=True to ignore single-image batches.

        source_valLoader = data.DataLoader(
            val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=2,
              pin_memory=True,drop_last=True,worker_init_fn=worker_init_fn)
        
        target_valLoader = data.DataLoader(
            val_dst_t, batch_size=opts.batch_size, shuffle=False, 
            num_workers=2, pin_memory=True,drop_last=True,worker_init_fn=worker_init_fn)
        
        print("Dataset: %s, Train set: %d, Val set: %d, Val set Target: %d" %
            (opts.dataset, len(train_dst), len(val_dst),len(val_dst_t)))
    

    # Set up model (all models are 'constructed at network.modeling)
    model = deeplab.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride,pretrained_backbone=True)
    if opts.separable_conv and 'plus' in opts.model:
        deeplab.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    #metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], momentum=0.9, weight_decay=opts.weight_decay)
    
    scaler = GradScaler() 

    class_weights=[0.0006, 0.0037, 0.0010, 0.0341, 0.0255, 0.0182, 0.1076, 0.0406, 0.0014,
        0.0193, 0.0056, 0.0183, 0.1655, 0.0032, 0.0836, 0.0951, 0.0960, 0.2267,
        0.0540]
    
    weights = torch.FloatTensor(class_weights).to(device)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean',weight=weights).to(device)

    #EarlyStop
    #early_stopping=EarlyStopping(patience=1000, delta=0)

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None: #and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model = nn.DataParallel(model)
        model.module.load_state_dict(checkpoint["model_state"])
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#

    if opts.test_only:
        model.eval()
        test_score=validate(opts=opts, model=model, loader=target_testLoader, device=device)
        return

    #W&B 
    wandb.login(key="***")
    
    with wandb.init(project="Paper", name=opts.run_name):
        interval_loss = 0
        total_it=0

        for e in range(opts.num_epoch):  
            # =====  Train  =====
            model.train()
            cur_itrs = 0

            for (images, labels) in source_trainLoader:
                cur_itrs += 1
                total_it+=1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                np_loss = loss.detach()  
                interval_loss += np_loss
                    
                if (total_it) % opts.val_interval == 0:
                    interval_loss = interval_loss / opts.val_interval

                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                            (e,  total_it, opts.total_itrs,  interval_loss))

                    save_ckpt('checkpoints/latest_%s.pth' %
                            ( opts.run_name))
                    print("validation...")
                    model.eval()
                    
                    print("SOURCE")
                    val_score_s = validate(opts=opts, model=model, loader=source_valLoader, device=device)
                    
                    print("\nTARGET")
                    val_score_t= validate(opts=opts, model=model, loader=target_valLoader, device=device)

                    if val_score_s['Mean IOU'] > best_score:  # save best model
                        best_score = val_score_s['Mean IOU']
                        save_ckpt('checkpoints/best_%s.pth' %
                                ( opts.run_name))            

                    if opts.dataset == 'cityscapes':
                        wandb.log({"mean_IOU_s": val_score_t['Mean IOU'],
                                    "mean_acc_s":val_score_t['Mean ACC'],
                                    "overall_acc_s": val_score_t['Overall ACC'],
                                    #"overall_iou_s": val_score_t['Overall IOU'],

                                    "mean_IOU_t": val_score_s['Mean IOU'],
                                    "mean_acc_t":val_score_s['Mean ACC'],
                                    "overall_acc_t": val_score_s['Overall ACC'],
                                    #"overall_iou_t": val_score_s['Overall IOU'],
                                    
                                    "interval_loss": interval_loss
                                    })
                    else:
                        wandb.log({"mean_IOU_s": val_score_s['Mean IOU'],
                                    "mean_acc_s":val_score_s['Mean ACC'],
                                    "overall_acc_s": val_score_s['Overall ACC'],
                                    #"overall_iou_s": val_score_s['Overall IOU'],

                                    "mean_IOU_t": val_score_t['Mean IOU'],
                                    "mean_acc_t":val_score_t['Mean ACC'],
                                    "overall_acc_t": val_score_t['Overall ACC'],
                                    #"overall_iou_t": val_score_t['Overall IOU'],

                                    "interval_loss": interval_loss
                                    })

                    interval_loss = 0.0

                    torch.cuda.empty_cache() 
                    model.train()
                    
                scheduler.step()

                if opts.itrs_goal is not None and total_it==opts.itrs_goal:
                    return

                if total_it >=  opts.total_itrs:
                    return
                
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    main()