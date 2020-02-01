from neuralnets import *
from PIL import Image
import torchvision.transforms as transforms
import torch

class SmileClassifier():
    def __init__(self,nnetwork:str="lenet",gpu:int=-1):
        if nnetwork == "lenet":
            Model = LeNet
            self.dims = (28,28)
        else:
            ValueError(f"Given {nnetwork} Network Not Exists")
        
        self.labels = ("neutral","smiling")
        self._device = torch.device("cpu") if gpu==-1 else torch.device(f"cuda:{gpu}")
        self.gpu_enabled = gpu != -1
        self.nnetwork = Model(train=False,class_size=2,model_path=f"./models/{nnetwork}/best.pt")
        self.to_pil = transforms.ToPILImage()
        self.to_gray = transforms.Grayscale(num_output_channels=1)
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.dims),
            transforms.ToTensor()
        ])

    def preprocess(self,data):
        if isinstance(data,str):
            data = Image.open(data)
            data = self.to_gray(data)
            data = self.to_tensor(data)
            if len(list(data.size())) == 3:
                data = torch.unsqueeze(data,dim=0)
            return data
        elif isinstance(data,np.ndarray): # assumed RGB or gray
            data = self.to_pil(data)
            data = self.to_tensor(data)
            if len(list(data.size())) == 3:
                data = torch.unsqueeze(data,dim=0)
            return data
        elif isinstance(data,torch.Tensor):
            if len(list(data.size())) == 3:
                data = torch.unsqueeze(data,dim=0)
            return data
        else:
            ValueError("Data in wrong format")

    def __call__(self,data,**kwargs):
        data = self.preprocess(data)
        with torch.no_grad():
            preds = self.nnetwork(data)
        if self.gpu_enabled:
            preds = preds.cpu()
        
        preds = preds.numpy()
        results = []
        for pred in preds:
            results.append({
                "label": self.labels[pred.argmax()],
                "score": pred.max()
            })
        return results

        
        
if __name__ == '__main__':
    import sys
    import cv2
    sc = SmileClassifier(nnetwork="lenet")
    
    bgr = cv2.imread(sys.argv[1])
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    print(sc(gray))
    """
    print(sc(sys.argv[1]))
    """