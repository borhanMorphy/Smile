from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import os
import random

class SmileDataset(Dataset):
	def __init__(self, 
			root_folder:str="./data",accepted_formats:list=["jpg","png","jpeg"]):
		super(SmileDataset,self).__init__()
		negative_folder = os.path.join(root_folder,"negatives")
		positive_folder = os.path.join(root_folder,"positives")
		self.labels = ("negative","positive") # 0,1

		self.transform = transforms.Compose(
			[
				transforms.Resize((28,28)),
				transforms.ToTensor()
			]
		)
		self.cls_counts = {}

		[self.cls_counts.update({i:1.0}) for i in range(len(self.labels))]
		
		self.data = [
			self.load_data(os.path.join(negative_folder,im_path),"negative") 
			for im_path in os.listdir(negative_folder)
			if im_path.split(".")[-1].lower() in accepted_formats]
		
		self.data += [
			self.load_data(os.path.join(positive_folder,im_path),"positive")
			for im_path in os.listdir(positive_folder)
			if im_path.split(".")[-1].lower() in accepted_formats]

		random.shuffle(self.data)

	def load_data(self,img_path:str,label:str):
		# count label
		self.cls_counts[self.labels.index(label)] += 1

		# load image as PIL Image
		img = Image.open(img_path)

		# (resize => normalize => to tensor) transformations
		img = self.transform(img)

		# convert label to tensor
		label = self.label_to_tensor(label)
		return img,label

	def label_to_tensor(self,label):
		return torch.tensor(self.labels.index(label))

	def __len__(self):
		return len(self.data)

	def __getitem__(self,idx):
		# select indexed data
		return self.data[idx]
