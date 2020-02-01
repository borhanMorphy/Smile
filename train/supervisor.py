import torch.nn as nn
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torch.optim as optim
import os

class SuperVisor:
	def __init__(self,model:nn.Module, dataset:Dataset,
		train_cfg:dict={},gpu:int=-1):
		
		self._train_cfg = self.get_train_cfg()
		self._train_cfg.update(train_cfg)
		self._device = torch.device("cpu") if gpu == -1 else torch.device(f"cuda:{gpu}")
		
		self._model = model
		self._model.train()
		self._model.to(self._device)

		self._dataset = dataset
		
	def get_data_loaders(self):
		data_split = self._train_cfg.get("data_split")
		batch_size = self._train_cfg.get("batch_size")
		num_workers = self._train_cfg.get("num_workers")
		assert "train" in data_split and "test" in data_split,"data split must contain train and test"
		sizes = {}
		total_size = len(self._dataset)
		current_size = total_size
		for key,perc in data_split.items():
			sizes[key] = int(total_size*perc)
			current_size -= sizes[key]
		
		assert current_size >= 0,"data split failed"
		
		if current_size > 0: sizes["train"] += current_size
		
		dls = {}
		iter_sizes = {}
		for key,ds in zip(sizes.keys(),random_split(self._dataset,sizes.values())):
			dls[key] = DataLoader(ds,
				batch_size=batch_size,
				shuffle= key == "train",
				num_workers=num_workers if key == "train" else 1)
			
			iter_sizes[key] = int(len(ds)//batch_size)
			if len(ds) % batch_size != 0:
				iter_sizes[key] += 1
		return dls,iter_sizes

	def get_train_cfg(self):
		if not self.__dict__.get("_train_cfg"):
			return {
				"model_name":"unknown",
				"epochs":10,
				"learning_rate":1e-2,
				"momentum":.9,
				"batch_size":16,
				"output_path":"./models",
				"class_size":2,
				"num_workers":1,
				"data_split":{
					"train":0.8,
					"test":0.2
				},
				"verbose":10 # every 10 batch, show train loss
			}
		return self._train_cfg

	def save_model(self, model_name, output_path):
		save_path = os.path.join(output_path,model_name)
		torch.save(self._model.state_dict(),save_path)


	def train(self, loss_fn:callable=nn.CrossEntropyLoss(), callback:callable=None):
		dls,iter_sizes = self.get_data_loaders()

		learning_rate = self._train_cfg.get("learning_rate")
		momentum = self._train_cfg.get("momentum")
		epochs = self._train_cfg.get("epochs")
		output_path = self._train_cfg.get("output_path")
		verbose = self._train_cfg.get("verbose")
		model_name = self._train_cfg.get("model_name")
		output_path = self._train_cfg.get("output_path")
		output_path = os.path.join(output_path,model_name)
		best_score = 1e+3

		if not os.path.isdir(output_path): os.makedirs(output_path)

		optimizer = optim.SGD(self._model.parameters(),
			lr=learning_rate,momentum=momentum)

		for epoch in range(epochs):
			running_loss = .0
			for i,(batchX,batchY) in enumerate(dls.get("train")):
				# set parameters grads to zero
				optimizer.zero_grad()
				
				# load to defined device
				batchX = batchX.to(self._device)
				batchY = batchY.to(self._device)

				# forward prop
				batchX = self._model(batchX)

				# calculate loss
				loss = loss_fn(batchX,batchY)

				# backward prop
				# TODO apply weighted loss to handle inbalanced dataset
				loss.backward()

				# optimize
				optimizer.step()

				# log statistics
				running_loss += loss.item()
				if (i+1) % verbose == 0:
					print("Training Info:")
					print(f"\tEpoch [{epoch+1}/{epochs}]\tIter [{i+1}/{iter_sizes.get('train')}]\tLoss:{running_loss/verbose:.4f}")
					running_loss = .0
			
			for key,dl in dls.items():
				if key == "train":
					continue
				loss = .0
				for batchX,batchY in dl:
					# load to defined device
					batchX = batchX.to(self._device)
					batchY = batchY.to(self._device)

					batchX = self._model(batchX)
					loss += loss_fn(batchX,batchY).item()
				loss = loss/iter_sizes.get(key)
				print(f"Epoch [{epoch+1}/{epochs}] for {key} loss:{loss:.4f}")
				if key == "test":
					if loss < best_score:
						self.save_model("best.pt",output_path)
						best_score = loss
					
					self.save_model("last.pt",output_path)



def save_checkpoint(my_model,model_name,train_loss,test_loss,epoch,out_dir='./checkpoints'):
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)
		
	torch.save(
		my_model.state_dict(),
		os.path.join(out_dir,f"{model_name}_{train_loss:.3f}_{test_loss:.3f}_{epoch}.pt"))