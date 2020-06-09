import argparse
import glob, os
import torch
import sys
import time
import torch.nn as nn
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from dataloader import SaliconDataset
from loss import *
import cv2
from utils import blur, AverageMeter
import sys
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--no_epochs',default=30, type=int)
parser.add_argument('--lr',default=1e-4, type=float)
parser.add_argument('--kldiv',default=True, type=bool)
parser.add_argument('--cc',default=True, type=bool)
parser.add_argument('--sim',default=True, type=bool)
parser.add_argument('--lr_sched',default=False, type=bool)
parser.add_argument('--enc_model',default="vggssm", type=str)
parser.add_argument('--optim',default="Adam", type=str)
#parser.add_argument('--load_weight',default=1, type=int)
parser.add_argument('--kldiv_coeff',default=10.0, type=float)
parser.add_argument('--step_size',default=3, type=int)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=-1.0, type=float)
parser.add_argument('--dataset_dir',default="/home/caoge/dataset/SalGAN/", type=str)
parser.add_argument('--batch_size',default=24, type=int)
parser.add_argument('--log_interval',default=60, type=int)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--model_val_path',default="model-vggssm-combinedloss.pkl", type=str)


args = parser.parse_args()

train_img_dir = args.dataset_dir + "images/train/"
train_gt_dir = args.dataset_dir + "maps/train/"
train_fix_dir = args.dataset_dir + "fixations/train/"

val_img_dir = args.dataset_dir + "images/val/"
val_gt_dir = args.dataset_dir + "maps/val/"
val_fix_dir = args.dataset_dir + "fixations/val/"

if args.enc_model == "vggm":
    print("VGGM")
    from VGGM import VGGM
    model = VGGM(3)

elif args.enc_model == "vggm+self-attention":
    print("vgg+self-attention")
    from VGGSAM import VGGSAM
    model = VGGSAM(3)

elif args.enc_model == "vggssm":
    print("VGGSSM")
    from VGGSSM import VGGSSM
    model = VGGSSM(3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
	print("Let's use", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
model.to(device)

train_img_ids = [nm.split(".")[0] for nm in os.listdir(train_img_dir)]  #列出路径下所有文件名称（不包括该文件的所有路径）
val_img_ids = [nm.split(".")[0] for nm in os.listdir(val_img_dir)]

train_dataset = SaliconDataset(train_img_dir, train_gt_dir, train_fix_dir, train_img_ids)
val_dataset = SaliconDataset(val_img_dir, val_gt_dir, val_fix_dir, val_img_ids)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.no_workers)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)


def loss_func(pred_map, gt, args):
    loss = torch.FloatTensor([0.0]).cuda()
    criterion = nn.L1Loss()
    if args.kldiv:
        loss += args.kldiv_coeff * kldiv(pred_map, gt)
    if args.cc:
        loss += args.cc_coeff * cc(pred_map, gt)
    if args.sim:
        loss += args.sim_coeff * similarity(pred_map, gt)
    return loss

def train(model, optimizer, loader, epoch, device, args):
    model.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt) in enumerate(loader):
        img = img.to(device)    # [32,3,256,256]
        gt = gt.to(device)      #[32,256,256]
        # fixations = fixations.to(device)
        # print(fixations.shape)
        optimizer.zero_grad()
        #print('img',img.shape)
        pred_map = model(img)       #[8,256,256]
        assert pred_map.size() == gt.size()
        #loss = nn.BCELoss(pred_map,gt)
        loss = loss_func(pred_map, gt, args)
        #print('loss',loss)
        loss.backward()
        total_loss += loss.item()
        cur_loss += loss.item()
        
        optimizer.step()
        if idx%args.log_interval==(args.log_interval-1):
            print('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
            cur_loss = 0.0
            sys.stdout.flush()
    
    print('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)

def validate(model, loader, epoch, device, args):
    model.eval()
    tic = time.time()
    total_loss = 0.0
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    
    for (img, gt) in loader:
        img = img.to(device)
        gt = gt.to(device)
        pred_map = model(img)
        # Blurring
        blur_map = pred_map.cpu().squeeze(0).clone().numpy()
        blur_map = blur(blur_map).unsqueeze(0).to(device)
        
        cc_loss.update(cc(blur_map, gt))    
        kldiv_loss.update(kldiv(blur_map, gt))    
        nss_loss.update(nss(blur_map, gt))
        sim_loss.update(similarity(blur_map, gt))
    judge_loss = args.kldiv_coeff * kldiv_loss.avg + args.cc_coeff * cc_loss.avg + args.sim_coeff * sim_loss.avg
    #judge_loss = args.kldiv_coeff * kldiv_loss.avg + args.cc_coeff * cc_loss.avg 
    #judge_loss = args.kldiv_coeff * kldiv_loss.avg
    print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, sum : {:.5f},  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, judge_loss, (time.time()-tic)/60))
    sys.stdout.flush()
    
    return judge_loss

params = list(filter(lambda p: p.requires_grad, model.parameters())) 

if args.optim=="Adam":  #using Adam
    optimizer = torch.optim.Adam(params, lr=args.lr)
if args.optim=="Adagrad":
    optimizer = torch.optim.Adagrad(params, lr=args.lr)
if args.optim=="SGD":
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
if args.lr_sched:                                               # default is False
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

print(device)

for epoch in tqdm(range(0, args.no_epochs)):
    loss = train(model, optimizer, train_loader, epoch, device, args)
    with torch.no_grad():
        judge_loss = validate(model, val_loader, epoch, device, args)
        if epoch == 0 :
            best_loss = judge_loss
        if best_loss >= judge_loss:
            best_loss = judge_loss
            print('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
            if torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), args.model_val_path)
            else:
                torch.save(model.state_dict(), args.model_val_path)
        print()

    if args.lr_sched:
        scheduler.step()
