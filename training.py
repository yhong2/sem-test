#training.py
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import LG_1d
import argparse
import scipy as sp
from scipy.sparse import diags
import gc
import subprocess, os
import net.network as network
from net.data_loader import *
from sem.sem import *
from plotting import plotter
from reconstruct import *


gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser("SEM")
parser.add_argument("--file", type=str, default='1000N31')
parser.add_argument("--batch", type=int, default=1000)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--ks", type=int, default=7)
args = parser.parse_args()

KERNEL_SIZE = args.ks
PADDING = (args.ks - 1)//2
FILE = args.file
SHAPE = int(args.file.split('N')[1]) + 1
BATCH = int(args.file.split('N')[0])
N, D_in, Filters, D_out = BATCH, 1, 32, SHAPE

xx = legslbndm(D_out)
lepolys = gen_lepolys(D_out, xx)
lepoly_x = dx(D_out, xx, lepolys)
lepoly_xx = dxx(D_out, xx, lepolys)

# Check if CUDA is available and then use it.
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)  
def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

# Load the dataset
try:
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
except:
	subprocess.call(f'python create_train_data.py --size {BATCH} --N {SHAPE - 1}', shell=True)
	lg_dataset = LGDataset(pickle_file=FILE, shape=SHAPE, subsample=D_out)
#Batch DataLoader with shuffle
trainloader = torch.utils.data.DataLoader(lg_dataset, batch_size=N, shuffle=True)
# Construct our model by instantiating the class
model1 = network.Net(D_in, Filters, D_out, kernel_size=KERNEL_SIZE, padding=PADDING)

# XAVIER INITIALIZATION
model1.apply(weights_init)
# SEND TO GPU
model1.to(device)
# Construct our loss function and an Optimizer.
criterion1 = torch.nn.L1Loss()
criterion2 = torch.nn.MSELoss(reduction="sum")
optimizer1 = torch.optim.LBFGS(model1.parameters(), history_size=10, tolerance_grad=1e-16, tolerance_change=1e-16, max_eval=20)


EPOCHS = args.epochs + 1
BEST_LOSS = 9E32
for epoch in tqdm(range(1, EPOCHS)):
	for batch_idx, sample_batch in enumerate(trainloader):
		f = Variable(sample_batch['f']).to(device)
		a = Variable(sample_batch['a']).to(device)
		u = Variable(sample_batch['u']).to(device)
		"""
		f -> alphas -> ?u
		"""
		def closure(f, a, u):
			if torch.is_grad_enabled():
				optimizer1.zero_grad()
			a_pred = model1(f)
			a = a.reshape(N, D_out-2)
			assert a_pred.shape == a.shape
			"""
			RECONSTRUCT SOLUTIONS
			"""
			u_pred = reconstruct(N, a_pred, lepolys)
			u = u.reshape(N, D_out)
			assert u_pred.shape == u.shape
			"""
			RECONSTRUCT ODE
			"""
			DE = ODE2(1E-1, u_pred, a_pred, lepolys, lepoly_x, lepoly_xx)
			f = f.reshape(N, D_out)
			assert DE.shape == f.shape
			"""
			COMPUTE LOSS
			"""
			loss1 = criterion2(a_pred, a) + criterion1(u_pred, u) + criterion1(DE, f)			
			if loss1.requires_grad:
				loss1.backward()
			return a_pred, u_pred, DE, loss1
		a_pred, u_pred, DE, loss1 = closure(f, a, u)
		optimizer1.step(loss1.item)
		current_loss = np.round(float(loss1.to('cpu').detach()), 6) 
	print(f"\tLoss1: {current_loss}")
	if epoch % 100 == 0 and 0 <= epoch < EPOCHS:
		plotter(xx, sample_batch, epoch, a=a_pred, u=u_pred, DE=DE)
	if current_loss < BEST_LOSS:
		torch.save(model1.state_dict(), f'model.pt')
		BEST_LOSS = current_loss

subprocess.call(f'python evaluate.py --file 100N{SHAPE-1} --ks {KERNEL_SIZE}', shell=True)
gc.collect()
torch.cuda.empty_cache()