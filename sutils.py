from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class AverageMeter(object): 
    """Computes and stores the average and current value""" 
    def __init__(self): 
        self.reset() 
    def reset(self): 
        self.val = 0 
        self.avg = 0 
        self.sum = 0 
        self.count = 0 
    def update(self, val, n=1): 
        self.val = val 
        self.sum += val * n 
        self.count += n
        self.avg = self.sum / self.count 

def accuracy(output, target, topk=(1,)): 
    """Computes the precision@k for the specified values of k""" 
    maxk = max(topk) 
    batch_size = target.size(0) 
    _, pred = output.topk(maxk, 1, True, True) 
    pred = pred.t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred)) 
    res = [] 
    for k in topk: 
        correct_k = correct[:k].view(-1).float().sum(0) 
        res.append(correct_k.mul_(100.0 / batch_size)) 
    return res 

def train(net, train_data, epoch, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    
    train_loss = 0
    train_acc = 0
    net = net.train()
    for im, label in train_data:
        if torch.cuda.is_available():
            im_train = Variable(im.cuda())  # (bs, 3, h, w)
            label_train = Variable(label.cuda())  # (bs, h, w)
        else:
            im_train = Variable(im)
            label_train = Variable(label)
        # forward
        output = net(im_train)
        loss = criterion(output, label_train) 
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += get_acc(output, label_train)

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
     
    epoch_str = ("Epoch %d. Train Loss: %f, Train Acc: %f, " %
    (epoch, train_loss / len(train_data),
    train_acc / len(train_data)))
    
    prev_time = cur_time
    print(epoch_str + time_str)




def validate_random(net, valid_data_random, epoch, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    
    valid_loss_random = 0
    valid_acc_random = 0
    net = net.eval()

    if  valid_data_random is not None:
        # random crop
        for im, label in valid_data_random:
            if torch.cuda.is_available():
                im_val = Variable(im.cuda(), volatile=True)
                label_val = Variable(label.cuda(), volatile=True)
            else:
                im_val = Variable(im, volatile=True)
                label_val = Variable(label, volatile=True)
            output_random = net(im_val)
            loss_random = criterion(output_random, label_val)
            valid_loss_random += loss_random.item()
            
            valid_acc_random += get_acc(output_random, label_val)

                
    epoch_str = (
        "Epoch %d. Random val Loss: %f, Random Acc: %f,"
        % (epoch, valid_loss_random / len(valid_data_random),
          valid_acc_random / len(valid_data_random)))
            
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    prev_time = cur_time
    print(epoch_str + time_str)

def validate_resize(net, valid_data_resize, epoch, optimizer, criterion):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    
    valid_loss_resize = 0
    valid_acc_resize = 0
    net = net.eval()

    if  valid_data_resize is not None:
        # random crop
        for im, label in valid_data_resize:
            if torch.cuda.is_available():
                im_val = Variable(im.cuda(), volatile=True)
                label_val = Variable(label.cuda(), volatile=True)
            else:
                im_val = Variable(im, volatile=True)
                label_val = Variable(label, volatile=True)
            output_resize = net(im_val)
            loss_resize = criterion(output_resize, label_val)
            valid_loss_resize += loss_resize.item()
            valid_acc_resize += get_acc(output_resize, label_val)

                
    epoch_str = (
        "Epoch %d. Resize val Loss: %f, Resize Acc: %f,"
        % (epoch, valid_loss_resize / len(valid_data_resize),
          valid_acc_resize / len(valid_data_resize)))
            
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    prev_time = cur_time
    print(epoch_str + time_str)
        
        
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(
        in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)

