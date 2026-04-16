#from tqdm import tqdm
import syn_to_cs.dann_syn as dann_syn
import utils
#import os
import random
import argparse
import numpy as np
from torch.utils import data
from dataload.cityscapes_syn import Cityscapes
from utils import ext_transforms as et
#from metrics import StreamSegMetrics
from torch.utils.data import random_split
import torch
import torch.nn as nn
from dataload.custom_dataset_syn import CustomDataset
#from sklearn.model_selection import train_test_split
from torchmetrics import JaccardIndex , Accuracy
import wandb
#from earlyStopping import EarlyStopping
from torch.cuda.amp import autocast, GradScaler

torch.cuda.empty_cache()

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes','gta','dann'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    parser.add_argument("--num_epoch", type=int, default=16)

    # Deeplab Options
    available_models = sorted(name for name in dann_syn.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              dann_syn.modeling.__dict__[name])
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

    parser.add_argument("--run_name", type=str, help="wandb run name")
    parser.add_argument("--alpha_weight", type=float, help="alpha weight",default=0.01)
    parser.add_argument("--txt", type=str, help="grad txt")
    parser.add_argument("--k", type=int, help="constant")


    parser.add_argument("--deterministic", action='store_true', default=False)

    parser.add_argument("--crop_size_h", type=int, help="Desired height of the output image.")
    parser.add_argument("--crop_size_w", type=int, help="Desired width of the output image.")

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
            et.ExtRandomCrop(size=(opts.crop_size_h, opts.crop_size_w)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    val_transform = et.ExtCompose([
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
    train_set_gta= CustomDataset(image_folder= opts.data_root + "/synthia/train/images/", label_folder=opts.data_root+ "/synthia/train/labels/")
    val_set_gta= CustomDataset(image_folder= opts.data_root + "/synthia/test/images/", label_folder=opts.data_root+ "/synthia/test/labels/")

    #transform
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

    if opts.test_only:
        city_test= Cityscapes(root=opts.data_root + "/cityscapes/",
                                split='val')
        city_test = TransformDataset(city_test, transform=test_transform)
        return  city_test
    else:
        return gta_train, gta_val, cs_train, cs_val     
                     



def validate(opts, model, loader, device, alpha):
    """Do validation and return specified samples"""
    #metrics.reset()

    accuracy_overall = Accuracy(task="multiclass", num_classes=opts.num_classes,average="micro",ignore_index=255).to(device)
    jaccard_overall = JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="micro",ignore_index=255).to(device)
    accuracy_mean = Accuracy(task="multiclass", num_classes=opts.num_classes,average="macro",ignore_index=255).to(device)
    jaccard_mean= JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="macro",ignore_index=255).to(device)
    accuracy_cls = Accuracy(task="multiclass", num_classes=opts.num_classes,average="none",ignore_index=255).to(device)
    jaccard_cls= JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="none",ignore_index=255).to(device)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs,_ = model(images,alpha)
            preds = outputs.detach().max(dim=1)[1]
            
            
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
            "Class-wise ACC": accuracy_cls.compute()
        }

        score_copy=score.copy()

        # Print results
        for metric, value in score.items():
            print(f"{metric}: {value}")

        # Reset metrics for future use
        for metric in [jaccard_overall, accuracy_overall, jaccard_mean, accuracy_mean, jaccard_cls, accuracy_cls]:
            metric.reset()

    return score_copy



def validate_test(opts, model, loader, device, alpha):
    """Do validation and return specified samples"""

    accuracy_overall = Accuracy(task="multiclass", num_classes=opts.num_classes,average="micro",ignore_index=255).to(device)
    jaccard_overall = JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="micro",ignore_index=255).to(device)
    accuracy_mean = Accuracy(task="multiclass", num_classes=opts.num_classes,average="macro",ignore_index=255).to(device)
    jaccard_mean= JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="macro",ignore_index=255).to(device)
    accuracy_cls = Accuracy(task="multiclass", num_classes=opts.num_classes,average="none",ignore_index=255).to(device)
    jaccard_cls= JaccardIndex(task="multiclass",num_classes=opts.num_classes,average="none",ignore_index=255).to(device)
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs,_ = model(images,alpha)
            preds = outputs.detach().max(dim=1)[1]
            
            
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
            "Class-wise ACC": accuracy_cls.compute()
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

def freeze_bn_stats(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()          # use running stats, stop updating them
            m.weight.requires_grad_(True)   # still learn gamma/beta
            m.bias.requires_grad_(True)

def keep_bn_in_eval_mode(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()

def main():
    opts = get_argparser().parse_args()

    opts.num_classes = 16


    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
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
        target_test = get_dataset(opts)
        target_testLoader = data.DataLoader(
            target_test, batch_size=opts.val_batch_size, shuffle=False, 
            num_workers= 1,pin_memory=True,drop_last=True,worker_init_fn=worker_init_fn)
        
        print("Target Test: %d" % (len(target_test)))
    else:
        source_train, source_val, target_train, target_val  = get_dataset(opts)

        source_trainLoader = data.DataLoader(
            source_train, batch_size=opts.batch_size, shuffle=True, num_workers=4,
            pin_memory=True,drop_last=True,worker_init_fn=worker_init_fn)  # drop_last=True to ignore single-image batches.

        source_valLoader = data.DataLoader(
            source_val, batch_size=opts.batch_size, shuffle=False, 
            num_workers= 4,pin_memory=True,drop_last=True,worker_init_fn=worker_init_fn)

        target_trainLoader = data.DataLoader(
            target_train, batch_size=opts.batch_size, shuffle=True, 
            num_workers=4,pin_memory=True, drop_last=True,worker_init_fn=worker_init_fn)  # drop_last=True to ignore single-image batches.

        target_valLoader = data.DataLoader(
            target_val, batch_size=opts.batch_size, shuffle=False,
              num_workers= 4,pin_memory=True,drop_last=True,worker_init_fn=worker_init_fn)
            
        print("Dataset: %s, Source Train set : %d, Source Val set: %d, Target Train set : %d, Target Val set : %d" %
            (opts.dataset, len(source_train), len(source_val),len(target_train), len(target_val)))

    # Set up model (all models are 'constructed at network.modeling)
    model = dann_syn.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        dann_syn.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    #metrics = StreamSegMetrics(opts.num_classes)

    classifier_optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1*0.01},
        {'params': model.classifier.parameters(), 'lr': 0.01},
    ], momentum=0.9, weight_decay=opts.weight_decay)

    # Set up optimizer
    discriminator_optimizer = torch.optim.AdamW(params=[
        {'params': model.discriminator.parameters(), 'lr': 0.01}
    ],  weight_decay=opts.weight_decay)

    scaler = GradScaler() 

    class_weights=[0.0010, 0.0009, 0.0006, 0.0641, 0.0655, 0.0168, 0.4513, 0.1720, 0.0017,
        0.0026, 0.0041, 0.0376, 0.0043, 0.0117, 0.0859, 0.0800]

    weights = torch.FloatTensor(class_weights).to(device)
    
    if opts.lr_policy == 'poly':
        classifier_scheduler = utils.PolyLR(classifier_optimizer, opts.total_itrs, power=0.9)
        discriminator_scheduler = utils.PolyLR(discriminator_optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        classifier_scheduler = torch.optim.lr_scheduler.StepLR(classifier_optimizer, step_size=opts.step_size, gamma=0.1)
        discriminator_scheduler = torch.optim.lr_scheduler.StepLR(discriminator_optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean',weight=weights).to(device)

    criterion_disc=nn.BCEWithLogitsLoss().to(device)

    #EarlyStop
    #early_stopping=EarlyStopping(patience=100, delta=0)


    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "classifier_optimizer": classifier_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "discriminator_scheduler": discriminator_scheduler.state_dict(),
            "classifier_scheduler": classifier_scheduler.state_dict(),
            "best_score_target": best_score_target,
            "best_score_source": best_score_source
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')

    # Restore

    best_score_target = 0.0
    best_score_source = 0.0
    
    if opts.ckpt is not None: #and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt,map_location=torch.device('cpu'))
        model = nn.DataParallel(model)
        model.module.load_state_dict(checkpoint["model_state"],strict=False)
        model.to(device)
        if opts.continue_training:
            discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])
            discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])
            classifier_scheduler.load_state_dict(checkpoint["classifier_scheduler"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score_target = checkpoint['best_score_target']
            best_score_source = checkpoint['best_score_source']
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
        test_score  = validate_test(
            opts=opts, model=model, loader=target_testLoader, device=device, alpha=0)
        return


    len_dataloader = max(len(source_trainLoader), len(target_trainLoader))
    
    wandb.login(key="***")
    
    with wandb.init(project="Paper_syn", name=opts.run_name):
        
        total_err_s_clf = total_err_s_disc = total_err_t_disc = total_clf_loss = total_disc_loss = interval_loss = 0
        total_it=0
        total_domain_acc_s = total_domain_acc_t = 0

        for e in range(opts.num_epoch):
            source_iter = iter(source_trainLoader)
            target_iter = iter(target_trainLoader)

            model.train()
            keep_bn_in_eval_mode(model.module.backbone)

            cur_itrs=0
            
            while cur_itrs < len_dataloader:
                total_it+=1
                cur_itrs +=1

                #a1
                p = float(cur_itrs + e * len_dataloader) / opts.num_epoch / len_dataloader
                alpha = (2. / (1. + np.exp(-1*(opts.k) * p)) - 1)*opts.alpha_weight
               
                discriminator_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                
                try:
                    source_image, source_label = next(source_iter)
                except StopIteration:  # Reset source iterator if exhausted (unlikely since source is longer)
                    source_iter = iter(source_trainLoader)
                    source_image, source_label = next(source_iter)

                source_image = source_image.to(device, dtype=torch.float32)
                source_label = source_label.to(device, dtype=torch.long)

                with autocast(dtype=torch.bfloat16):
                    class_output, domain_output_s = model(source_image, alpha)
                    err_s_clf = criterion(class_output, source_label)
                    domain_label_s = torch.zeros((opts.batch_size , domain_output_s.size(2), domain_output_s.size(3))).float().to(device)
                    err_s_disc = criterion_disc(domain_output_s.squeeze(1), domain_label_s)

                with torch.no_grad():
                    preds_s = (domain_output_s.squeeze(1) > 0).long()
                    correct_s = (preds_s == domain_label_s.long()).float().mean().item()
                    total_domain_acc_s += correct_s

                ###TARGET###
                try:
                    target_image, _ = next(target_iter)
                except StopIteration:  # Reset target iterator if exhausted
                    target_iter = iter(target_trainLoader)
                    target_image, _  = next(target_iter)
                    
                target_image = target_image.to(device, dtype=torch.float32)

                with autocast(dtype=torch.bfloat16):
                    _, domain_output = model(target_image, alpha)
                    domain_label = torch.ones((opts.batch_size ,domain_output.size(2),domain_output.size(3))).float().to(device)
                    err_t_disc = criterion_disc(domain_output.squeeze(1), domain_label)

                with torch.no_grad():
                    preds_t = (domain_output.squeeze(1) > 0).long()
                    correct_t = (preds_t == domain_label.long()).float().mean().item()
                    total_domain_acc_t += correct_t
                            
                total_loss = err_s_clf + ((err_t_disc + err_s_disc)/2)

                scaler.scale(total_loss).backward()
                scaler.step(discriminator_optimizer)
                scaler.step(classifier_optimizer)
                scaler.update()

                total_err_s_clf += err_s_clf.item()
                total_err_s_disc += err_s_disc.item()
                total_err_t_disc += err_t_disc.item()

                total_clf_loss += err_s_clf.item() 
                total_disc_loss += (err_s_disc.item() + err_t_disc.item())/2 

                interval_loss += total_loss.item()

                if (total_it) % opts.val_interval== 0:  

                    interval_loss=interval_loss / opts.val_interval

                    print("Epoch %d,  Itrs %d/%d, Total Loss=%f, total_clf_loss= %f ,  total_disc_loss=%f" %
                              (e,  total_it, opts.total_itrs, interval_loss, total_clf_loss, total_disc_loss ))

                    save_ckpt('checkpoints/latest_%s.pth' %
                            ( opts.run_name))
                    
                    print("validation...")

                    model.eval()

                    print("SOURCE")
                    val_score_source = validate(
                        opts=opts, model=model, loader=source_valLoader, device=device,alpha=alpha)
                    print("TARGET")
                    val_score_target = validate(
                        opts=opts, model=model, loader=target_valLoader, device=device,alpha=alpha)
                    
                    if val_score_target['Mean IOU'] > best_score_target:  # save best model
                            best_score_target = val_score_target['Mean IOU']
                            save_ckpt('checkpoints/best_TARGET_%s.pth' %
                                        (opts.run_name))
                            
                    if val_score_source['Mean IOU'] > best_score_source:  # save best model
                            best_score_source = val_score_source['Mean IOU']
                            save_ckpt('checkpoints/best_SOURCE_%s.pth' %
                                        (opts.run_name))

                    

                    wandb.log({#"interval_loss": interval_loss, 
                            #"clf_loss_s": total_err_s_clf/opts.val_interval, 
                            #"disc_loss_s": total_err_s_disc/opts.val_interval,
                            #"disc_loss_t": total_err_t_disc/opts.val_interval,
                            "total_clf_loss": total_clf_loss/opts.val_interval, 
                            "total_disc_loss": total_disc_loss/opts.val_interval,
                            "alpha":alpha,

                            "domain_acc_s": total_domain_acc_s / opts.val_interval,
                            "domain_acc_t": total_domain_acc_t / opts.val_interval,                            

                            "mean_IOU_s": val_score_source['Mean IOU'],
                            "mean_acc_s":val_score_source['Mean ACC'],
                            "overall_acc_s": val_score_source['Overall ACC'],
                            #"overall_iou_s":val_score_source['Overall IOU'],

                            "mean_IOU_t": val_score_target['Mean IOU'],
                            "mean_acc_t":val_score_target['Mean ACC'],
                            "overall_acc_t": val_score_target['Overall ACC'],
                            #"overall_iou_t":val_score_target['Overall IOU']
                            })
 
                    
                    total_err_s_clf = total_err_s_disc = total_err_t_disc = total_clf_loss = total_disc_loss = interval_loss = 0
                    total_domain_acc_s = total_domain_acc_t = 0

                    torch.cuda.empty_cache() 

                    model.train()
                    keep_bn_in_eval_mode(model.module.backbone)                    
                    
                discriminator_scheduler.step()
                classifier_scheduler.step()

                if opts.itrs_goal is not None and total_it==opts.itrs_goal:
                    return

                if total_it >=  opts.total_itrs:
                    return
            
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    main()
