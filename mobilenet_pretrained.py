from torchvision import models
import torch
 
dir(models)
mobilenet = models.mobilenet(pretrained = True)
