import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pdb

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image

import matplotlib.pyplot as plt



cuda = torch.cuda.is_available()
if cuda:
	checkpoint = torch.load('res_checkpoint100')
else:
	# Load GPU model on CPU
	checkpoint = torch.load(resume_weights,
							map_location=lambda storage,
							loc: storage)

resnet = models.resnet152()
resnet.fc = nn.Sequential(nn.Linear(2048*100, 228),nn.Sigmoid())

resnet.load_state_dict(checkpoint['state_dict'])



batch_size = 32
use_gpu = torch.cuda.is_available()

data_train = json.load(open('/scratch/ms6771/train.json'))
df_train = pd.DataFrame.from_records(data_train["annotations"])
data_val = json.load(open('/scratch/ms6771/validation.json'))
df_val = pd.DataFrame.from_records(data_val["annotations"])

val_size = 2000
val_ = df_val[:val_size]


mlb = MultiLabelBinarizer()
mlb = mlb.fit(df_train['labelId'])

data_transform_normalize = transforms.Compose([
	transforms.RandomRotation(degrees=15),
	transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225])
])

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

validation_dataset = FashionDataset(root = '/scratch/ms6771/Image/validation',
									data = val_,
									mlb = mlb,
									transform = data_transform_normalize)

validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = batch_size, drop_last = True)


resnet.eval()

predictions = np.zeros((1,228))
for batch_idx, (images, labels) in enumerate(validation_loader):
	# if use_gpu:
	# 	images = Variable(images.float(), volatile = True).cuda()
	# 	labels = Variable(labels.float()).cuda()
	# else:
	images = Variable(images, volatile = True)
	labels = Variable(labels).float()
	# pdb.set_trace()
	outputs = resnet(images)
	# pdb.set_trace()
	predictions = np.vstack((predictions,np.asarray(outputs.data)))
	if batch_idx % 10 ==0:
		print(batch_idx)

print('saving output')
np.savetxt('validation_outputs.csv', predictions, delimiter=',')









