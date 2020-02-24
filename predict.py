import numpy as np
import os,time
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torchvision import transforms as tfs
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt

path ="./results"

modelpath = "./mobilenet_model_100e.pth.tar"


batch_size = 10
workers = 4
epochs = 1
print_freq = 100

valdir = path
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)
model = torch.load(modelpath)

# switch to evaluate mode
model.eval()
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().data[0]
    return num_correct / total

def evaluate(val_loader, model):
    for im, label in val_loader:
        if torch.cuda.is_available():
            im_val = Variable(im.cuda())  # (bs, 3, h, w)
            label_val = Variable(label.cuda())  # (bs, h, w)
        else:
            im_val = Variable(im)
            label_val = Variable(label)
        # compute output
        output = model(im_val)
        # measure accuracy and record loss
        val_acc +=get_acc(output,label_val)

        # measure elapsed time
        end = time.time()

        print(
                  'acc {acc.val:.3f}'.format(
                   acc = val_acc))
evaluate(val_loader,model)