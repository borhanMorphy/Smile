## Brief Explanation
Real time (CPU) face detection using opencv haar cascade and a CNN classifier to classify faces that smiles or not.<br/>
For classification LeNet5 used and impelented with pytorch<br/>

## Dataset
You can obtain the dataset [here](https://github.com/hromi/SMILEsmileD/tree/master/SMILEs)

## Traning

If you have a gpu than you can use like this to train the classifier
```
python3 train.py --batch_size 64 --epochs 15 --model_name lenet -g 0
```

To train on CPU
```
python3 train.py --batch_size 64 --epochs 15 --model_name lenet -g -1
```

There will be 2 model on the models/ directory<br/>
best.pt refers that model saved when best eval result on test set achived.<br/>
last.pt refers that model saved when last epoch runned.<br/>

## Inference

To run the detector and classifier use inference.py
```
python3 inference.py
```

This will cause to open webcam at index 0 and run the classifier with detector using best.pt

## References
[LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)<br/>
[Dataset](https://github.com/hromi/SMILEsmileD/tree/master/SMILEs)<br/>
[Special thanks to Adrian Rosebrock and his awesome book](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/)<br/>