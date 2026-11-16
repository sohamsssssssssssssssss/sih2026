"""Small U-Net and explicitly masked BCE+Dice loss for T15."""
import torch
from torch import nn
from torch.nn import functional as F


def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(3,b),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(3,b),nn.ReLU())


class UNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        widths=[12,24,48,96,192]
        self.down=nn.ModuleList([block(a,b) for a,b in zip([in_channels]+widths[:-1],widths)])
        self.up=nn.ModuleList([block(a+b,b) for a,b in zip(widths[:0:-1],widths[-2::-1])])
        self.head=nn.Conv2d(widths[0],1,1)

    def forward(self,x):
        skips=[]
        for i,layer in enumerate(self.down):
            if i:
                x=F.max_pool2d(x,2)
            x=layer(x);skips.append(x)
        for layer,skip in zip(self.up,skips[-2::-1]):
            x=F.interpolate(x,size=skip.shape[-2:],mode='nearest')
            x=layer(torch.cat([x,skip],1))
        return self.head(x)


def loss(logits,target,pos_weight=1.,kind="bce_dice"):
    if kind not in ('bce','bce_dice'):
        raise ValueError(f'Unknown loss {kind}')
    valid=target!=-1
    if not valid.any():
        return logits[valid].sum()
    z,y=logits[valid],target[valid]
    bce=F.binary_cross_entropy_with_logits(z,y,pos_weight=z.new_tensor(pos_weight))
    prob=torch.sigmoid(z)
    dice=1-(2*(prob*y).sum()+1)/(prob.sum()+y.sum()+1)
    return bce if kind=="bce" else bce+dice


class FullResolutionCNN(nn.Module):
    """Six 3x3 conv layers at native resolution, GroupNorm/ReLU, then a 1x1 head."""
    def __init__(self,in_channels):
        super().__init__()
        layers=[]
        for a in [in_channels]+[64]*5:
            layers.extend([nn.Conv2d(a,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU()])
        self.features=nn.Sequential(*layers)
        self.head=nn.Conv2d(64,1,1)

    def forward(self,x):
        return self.head(self.features(x))
