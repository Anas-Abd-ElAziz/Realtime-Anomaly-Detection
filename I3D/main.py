from pathlib import Path
import shutil
import argparse
import numpy as np
import time
import ffmpeg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from extract_features import run
from resnet import i3_res50
import os


def generate(frames, pretrainedpath, frequency, batch_size, sample_mode):
	i3d = i3_res50(400, pretrainedpath)
	if torch.cuda.is_available():
		i3d.cuda()
	i3d.train(False)  # Set model to evaluate mode
	startime = time.time()
	print("Preprocessing done..")
	features = run(frames, i3d, frequency, batch_size, sample_mode)
	print("Obtained features of size: ", features.shape)
	#shutil.rmtree(temppath)
	print("done in {0}.".format(time.time() - startime))
	return features

if __name__ == '__main__': 
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasetpath', type=str, default="samplevideos/")
	parser.add_argument('--outputpath', type=str, default="output")
	parser.add_argument('--pretrainedpath', type=str, default="pretrained/i3d_r50_kinetics.pth")
	parser.add_argument('--frequency', type=int, default=16)
	parser.add_argument('--batch_size', type=int, default=20)
	parser.add_argument('--sample_mode', type=str, default="oversample")
	args = parser.parse_args()
	generate(frames, args.pretrainedpath, args.frequency, args.batch_size, args.sample_mode)    
