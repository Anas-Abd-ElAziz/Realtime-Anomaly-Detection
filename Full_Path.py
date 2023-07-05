import cv2
import numpy as np
import glob
import os
from I3D import main as feat_ex
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *

root = ""

# set up everything for the testing (viz, loaders, model)
def set_up():

    args = option.parser.parse_args()
    config = Config(args)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                            batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint = torch.load(root + "ckpt/shanghai_best_ckpt.pkl", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint)
    return test_loader, model, args, device

################################################################
def anomaly(frames):
    features = feat_ex.generate(frames, root + "I3D/pretrained/i3d_r50_kinetics.pth",4,10,"oversample")
    test_loader, model, args, device = set_up()
    return test(test_loader, model, args, device, features)