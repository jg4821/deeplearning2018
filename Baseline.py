from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import time
import pdb

import io
import os
import sys
import json
# import urllib3
import multiprocessing

from PIL import Image
# from urllib3.util import Retry

import torch
import torchvision
import torch.utils.data as data_utils
from torchvision.transforms import ToTensor
from torchvision.transforms import Resize
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn


class FashionDataset(Dataset):
	def __init__(self, root, data, mlb, transform=None):
		self.root = os.path.expanduser(root)
		self.transform = transform
		self.X = None
		self.y = mlb.transform(data['labelId'])
	def __getitem__(self, index):
		img_path = os.path.join(self.root,"{}.jpg".format(index + 1))
		image = Image.open(img_path)
		label = self.y[index]
		if self.transform is not None:
			image = self.transform(image)
		return image, label
	def __len__(self):
		return self.y.shape[0]


# __CNN (Baseline)__

class CNNModule(nn.Module):
	def __init__(self):
		super (CNNModule,self).__init__()
		
		self.cnn1 = nn.Conv2d(in_channels = 3,out_channels = 30,kernel_size = 3,stride = 1)
		self.relu1 = nn.ReLU()
		nn.init.xavier_uniform(self.cnn1.weight)

		self.maxpool1=nn.MaxPool2d(kernel_size = 2)

		self.cnn2 = nn.Conv2d(in_channels = 30,out_channels = 90,kernel_size = 3,stride = 1)
		self.relu2 = nn.ReLU()
		nn.init.xavier_uniform(self.cnn2.weight)
		
		self.maxpool2 = nn.MaxPool2d(kernel_size = 2)

		self.fcl = nn.Linear(90*123*123,228)
		
		self.activation = nn.Sigmoid()
		
	def forward(self,x):
		out = self.cnn1(x)
		out = self.relu1(out)
#         print ("CNN1", out.size())
		out = self.maxpool1(out)
#         print ("Maxpool1", out.size())
		
		out = self.cnn2(out)
		out = self.relu2(out)
#         print ("CNN2", out.size())
		out = self.maxpool2(out)
#         print ("Maxpool2", out.size())
		# print(out.size())
		out = out.view(out.size(0),-1)

#         print ("Reshape", out.size())
		out = self.fcl(out)
#         print("Fully connected", out.size())
		raw_out = out
		out = self.activation(out)
#         print("Output", out.size())
		return out, raw_out

def squared_cross_entropy_loss(preds,labels):
	entropy = (-(
		labels * torch.log(preds+1e-8) + 
		(1-labels) * torch.log(1-preds+1e-8)
		))**(1.5)
	return torch.mean(entropy)


def train(epoch, model, optimizer, train_loader):
	model.train()
	for batch_idx, (images, labels) in enumerate(train_loader):
		# pdb.set_trace()
		if use_gpu:
			images = Variable(images.float()).cuda()
			labels = Variable(labels.float()).cuda()                    
		else:
			images = Variable(images, volatile = True)
			labels = Variable(labels).float()

		# pdb.set_trace()
		optimizer.zero_grad()
		outputs, raw_output = model(images)
		criterion = nn.BCELoss()
		# print(outputs.size(),labels.size())
		loss = criterion(outputs,labels)
		# loss = squared_cross_entropy_loss(outputs,labels)
		# print(epoch,loss[0])
		# print(outputs.mean(),outputs.std())
		#pdb.set_trace()
		loss.backward()

		optimizer.step()

		mean_out = outputs.data.mean()

		pred = outputs.data.gt(mean_out)
		tp = (pred + labels.data.byte()).eq(2).sum()
		fp = (pred - labels.data.byte()).eq(1).sum()
		fn = (pred - labels.data.byte()).eq(-1).sum()
		tn = (pred + labels.data.byte()).eq(0).sum()
		acc = (tp + tn) / (tp + tn + fp + fn)
		try:
			prec = tp / (tp + fp)
		except ZeroDivisionError:
			prec = 0.0
		try:
			rec = tp / (tp + fn)
		except ZeroDivisionError:
			rec = 0.0
		try:
			f1 = 2*(rec * prec) / (rec + prec)
		except ZeroDivisionError:
			f1 = 0.0

		if batch_idx % 100 == 0:    
			print('Train Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}'.format(
				epoch, batch_idx * len(images), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
				loss.data[0], acc, prec, rec, f1))



def validation(model, validation_loader):
	model.eval()
	validation_loss = 0
	acc = prec = rec = f1 = 0
	# pdb.set_trace()

	for batch_idx, (images, labels) in enumerate(validation_loader):
		if use_gpu:
			images = Variable(images.float(), volatile = True).cuda()
			labels = Variable(labels.float()).cuda()                    
		else:
			images = Variable(images, volatile = True)
			labels = Variable(labels).float()

		outputs, raw_output = model(images)
		criterion = nn.BCELoss()
		validation_loss = criterion(outputs,labels)
		# outputs, raw_output = model(images)
		# validation_loss = squared_cross_entropy_loss(outputs,labels)

		mean_out = outputs.data.mean()

		pred = outputs.data.gt(mean_out)
		tp = (pred + labels.data.byte()).eq(2).sum()
		fp = (pred - labels.data.byte()).eq(1).sum()
		fn = (pred - labels.data.byte()).eq(-1).sum()
		tn = (pred + labels.data.byte()).eq(0).sum()
		acc = (tp + tn) / (tp + tn + fp + fn)
		try:
			prec = tp / (tp + fp)
		except ZeroDivisionError:
			prec = 0.0
		try:
			rec = tp / (tp + fn)
		except ZeroDivisionError:
			rec = 0.0
		try:
			f1 = 2*(rec * prec) / (rec + prec)
		except ZeroDivisionError:
			f1 = 0.0

		# validation_loss /= len(validation_loader.dataset)
		if batch_idx % 10 == 0:
			print('\nValidation set: Batch {}: Batch loss: {:.6f} | Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f} | F1 Score: {:.4f}'.format(
				batch_idx, validation_loss.data[0], acc, prec, rec, f1))

def save_checkpoint(state, epoch, filename='bs_checkpoint'):
	print ("=> Saving model")
	torch.save(state, filename+str(epoch))


# __Main__
batch_size = 32
num_epoch = 10
use_gpu = torch.cuda.is_available()


# __Convert Data to Dataloader__
data_train = json.load(open('/scratch/ms6771/train.json'))
data_val = json.load(open('/scratch/ms6771/validation.json'))
df_train = pd.DataFrame.from_records(data_train["annotations"])
df_val = pd.DataFrame.from_records(data_val["annotations"])

train_size = 10000
val_size = 2000
train_ = df_train[:train_size]
val_ = df_val[:val_size]

# initialize multilabel binarizer
mlb = MultiLabelBinarizer()
mlb = mlb.fit(df_train['labelId'])

#Reload dataset with normalization
data_transform_normalize = transforms.Compose([
	transforms.RandomRotation(degrees=15),
	transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225])
	# transforms.Normalize(mean = train_mean.tolist(), std = train_std.tolist())
])


train_dataset = FashionDataset(root = '/scratch/ms6771/Image/train',
							   data = train_,
							   mlb = mlb,
							   transform = data_transform_normalize)

validation_dataset = FashionDataset(root = '/scratch/ms6771/Image/validation',
									data = val_,
									mlb = mlb,
									transform = data_transform_normalize)


train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, drop_last = True)


if use_gpu:
	model = CNNModule().cuda()
else:
	model = CNNModule()

# optimizer = torch.optim.SGD(model.parameters(),lr = 0.015)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-6)
# optimizer = torch.optim.Adam(model.parameters())


print('BASELINE MODEL')
for epoch in range(1, num_epoch + 1):
	start = time.time()
	train(epoch, model, optimizer, train_loader)
	end = time.time()
	print('Training Time: {} minutes'.format((end - start)/60))
	
	validation(model, validation_loader)

	if epoch %10==0:
		save_checkpoint({
			'epoch': epoch,
			'state_dict': model.state_dict()
			# 'best_accuracy': best_accuracy
		},epoch)




