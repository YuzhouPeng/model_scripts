import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Just some functions to average stuff, and save the model
from utils_pytorch import *

# Trainning parameters
learning_rate = 0.1
batch_size = 64
momentum = 0.9
weight_decay = 1e-4
workers = 4
print_freq = 100
epochs = 1

# Define your data path here
IMAGENET_PATH = '/home/yang/dataset/imagenet/tiny-imagenet-200'


# define mobilenet class
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        # Normal convolution block followed by Batchnorm (CONV_3x3-->BN-->Relu)
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # Depthwise convolution block (CONV_BLK_3x3-->BN-->Relu-->CONV_1x1-->BN-->Relu)
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


#Initialize model and pass to the gpu
model = MobileNet()

#print(model)

# use GPU to accelerate
model = torch.nn.DataParallel(model).cuda()

#define loss
criterion = nn.CrossEntropyLoss().cuda()

#define solver (sgd)
optimizer = torch.optim.SGD(model.parameters(),learning_rate,momentum=momentum,weight_decay=weight_decay)

#data loading specifics for ImageNet
# data loading code
traindir = os.path.join(IMAGENET_PATH,"train")
valdir = os.path.join(IMAGENET_PATH,"val")
normalize = transforms.Normalize(mean = [0.485,0,456,0,406],std = [0.229,0.224,0.225])

# preprocess the data, operations that will be done on data
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(traindir,transforms.Compose([
        transforms.RandomResizedCrop(224) #randomly crop image to 224*224
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(10), 
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), 
        transforms.ToTensor(), 
        normalize,
    ])),
    batch_size = batch_size,shuffle=True,
    num_workers = workers,pin_memory = True
    )

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=workers, pin_memory=True)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_size = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #switch to train mode
    model.train()

    end = time.time()
    for i, (input,target) in enumerate(train_loader):
        #measure data loading time
        data_time.update(time.time-end)

        target = target.cuda(async=True)
        input_val = torch.autograd.Variable(input)
        target_val = torch.autograd.Variable(target)

        #compute output
        output = model(input_val)
        loss =  criterion(output,target_val)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

#save model

#only save parameters
torch.save(model.state_dict(),"mobilenet_params.pth.tar")

#save whole model
torch.save(model,"mobilenet_model.pth.tar")