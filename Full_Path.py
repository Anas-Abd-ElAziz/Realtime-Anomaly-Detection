import cv2
import numpy as np
import glob
import os
import skvideo.io
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

'''
TO-DO : for @gemy 3shan y3rf m7tagen n3mel eh xd
        - Display Anomaly Score on Scree
        - Fix saving video with Yolo prediction
        - Seperate between anomaly prediction thread and displaying video thread
'''
root = ""

# set up everything for the testing (viz, loaders, model)
def set_up():
    # viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)

    args = option.parser.parse_args()
    config = Config(args)

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                            batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                            batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)

    #for name, value in model.named_parameters():
    #    print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    checkpoint = torch.load(root + "ckpt/shanghai_best_ckpt.pkl", map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    return test_loader, model, args, device

################################################################
def anomaly(frames):
    features = feat_ex.generate(frames, root + "I3D/pretrained/i3d_r50_kinetics.pth",4,10,"oversample")
    test_loader, model, args, device = set_up()
    return test(test_loader, model, args, device, features)