from dataset import SmileDataset
from neuralnets import LeNet # TODO: make it dynamic
from train.supervisor import SuperVisor

import torch.nn as nn
import torch

import argparse

def get_arguments():
	ap = argparse.ArgumentParser("Smile Classifier Training Module")
	ap.add_argument("-vv","--verbose",type=int,default=50)
	ap.add_argument("-b","--batch_size",type=int,default=32)
	ap.add_argument("-e","--epochs",type=int,default=10)
	ap.add_argument("-m","--model_name",type=str,default="unknown")
	ap.add_argument("-g","--gpu",type=int,default=-1,choices=[-1,0,1]) # use 0 or 1 to enable gpu

	return vars(ap.parse_args())

def get_cls_loss_weights(cls_counts):
	total_count = sum(cls_counts.values())
	max_value = max(cls_counts.values())
	cls_weights = [1 for _ in range(len(cls_counts.keys()))]
	for k,v in cls_counts.items():
		cls_weights[k] = max_value/v
	return cls_weights

def main(**kwargs):
	verbose = kwargs.get("verbose",50)
	batch_size = kwargs.get("batch_size",32)
	epochs = kwargs.get("epochs",10)
	model_name = kwargs.get("model_name","unknown")
	gpu = kwargs.get("gpu",-1)
	
	print("creating dataset...")
	ds = SmileDataset()
	
	print("creating model...")
	model = LeNet(train=True,class_size=2)

	print("creating supervisor instance for training...")
	train_cfg = dict(
		verbose=verbose,
		batch_size=batch_size,
		epochs=epochs,
		model_name=model_name)
	
	spv = SuperVisor(model,ds,gpu=gpu,train_cfg=train_cfg)

	cls_weights = get_cls_loss_weights(ds.cls_counts)
	loss_fn = nn.CrossEntropyLoss(torch.tensor(cls_weights).to(spv._device))

	print("starting to train...")
	
	spv.train(loss_fn=loss_fn)

	


if __name__ == '__main__':
	main(**get_arguments())