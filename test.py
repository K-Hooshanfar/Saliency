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
import numpy as np, cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from dataloader import TestLoader, SaliconDataset
from RSAM import RSAM
from loss import *
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--val_img_dir',default="/home/caoge/dataset/SalGAN/images/test/", type=str)
parser.add_argument('--model_val_path',default="model-vggssm-combinedloss.pkl", type=str)
parser.add_argument('--no_workers',default=4, type=int)
parser.add_argument('--enc_model',default="vggssm", type=str)
parser.add_argument('--results_dir',default="/home/caoge/Prediction_maps", type=str)
parser.add_argument('--validate',default=1, type=int)
parser.add_argument('--save_results',default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

model.load_state_dict(torch.load(args.model_val_path))
model = nn.DataParallel(model)
model = model.to(device)

val_img_ids = os.listdir(args.val_img_dir)
val_dataset = TestLoader(args.val_img_dir, val_img_ids)
vis_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.no_workers)

if args.save_results:
	visualize_model(model, vis_loader, device, args)