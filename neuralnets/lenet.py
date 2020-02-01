import torch.nn as nn
import torch
import numpy as np

class Flatten(nn.Module):
    def __init__(self,size):
        super(Flatten,self).__init__()
        self._size = size

    def forward(self,data):
        return data.view(-1,self._size)

class LeNet(nn.Module):
    """
    
    conv => relu => pool
    conv => relu => pool
    flatten => fc => relu
    fc => softmax    
    
    CONV    [Bx1x28x28 => Bx20x28x28]
    RELU    [Bx20x28x28 => Bx20x28x28]
    POOL    [Bx20x28x28 => Bx20x14x14]

    CONV    [Bx20x14x14 => Bx50x14x14]
    RELU    [Bx50x14x14 => Bx50x14x14]
    POOL    [Bx50x14x14 => Bx50x7x7]
    
    FLATTEN [Bx50x7x7 => Bx2450]
    FC      [Bx2450 => Bx500]
    RELU    [Bx500 => Bx500]

    FC      [Bx500 => Bx10]
    SOFTMAX [Bx10 => Bx10]
    """

    def __init__(self,
            train:bool=False,input_channel:int=1,
            class_size:int=10,model_path:str=None):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(
            input_channel,  # input channel
            20, # output channel
            kernel_size=(5,5),stride=1,padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0)
        self.dropout1 = nn.Dropout2d(p=.25)
        
        self.conv2 = nn.Conv2d(
            20, # input channel
            50, # output channel
            kernel_size=(5,5),stride=1,padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0)
        self.dropout2 = nn.Dropout2d(p=.25)

        self.flatten = Flatten(50*7*7)
        self.fc3 = nn.Linear(in_features=50*7*7,out_features=500)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=.25)

        self.fc4 = nn.Linear(in_features=500,out_features=class_size)
        self.softmax = nn.Softmax(dim=1)

        if model_path:
            print(f"[INFO] loading pretrained model:\n{model_path}")
            state_dict = torch.load(model_path)
            self.load_state_dict(state_dict)
        if train:
            self.train()
        else:
            self.eval()

    def forward(self,X):
        if isinstance(X,np.ndarray):
            X = torch.from_numpy(X)
        X = self.conv1(X)
        X = self.relu1(X)
        X = self.pool1(X)
        X = self.dropout1(X)

        X = self.conv2(X)
        X = self.relu2(X)
        X = self.pool2(X)
        X = self.dropout2(X)

        X = self.flatten(X)
        X = self.fc3(X)
        X = self.relu3(X)
        X = self.dropout3(X)

        X = self.fc4(X)
        return X if self.training else self.softmax(X)        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("lenet test")
    parser.add_argument("-m","--model_path",help="pre trained model path",type=str,default=None)
    parser.add_argument("-c","--class_size",help="class size",type=int,default=10)
    parser.add_argument("-t","--train",help="set true if traning",type=bool,default=False)
    args = parser.parse_args()
    
    model = LeNet(
        model_path=args.model_path,
        training=args.train,
        class_size=args.class_size)
    
    print(model)
