# Accuracy, Latency, training time, data points processes per sec

from horovod import torch as hvd

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader, distributed, TensorDataset
from torchvision import datasets, transforms, models
import torch.nn as nn
import os
import math
from tqdm import tqdm
import zipfile
import numpy as np
import pandas as pd
import numpy as np
import time


par={'nproc':1,'batch_size':8, 'epochs':1, 'cuda':torch.cuda.is_available(), 'model':'resnet50', 'dataset':'MNIST'}

class resnet50_mod(nn.Module):
  def __init__(self, in_channels=1):
    super(resnet50_mod, self).__init__()

    self.model=models.resnet50(pretrained=True)
    self.model.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.fc=nn.Linear(self.model.fc.in_features, 10)

  def forward(self, x):
    return self.model(x)

class vgg16_mod(nn.Module):
  def __init__(self, in_channels=1):
    super(vgg16_mod, self).__init__()

    self.model=models.vgg16(pretrained=True)
    self.model.conv1=nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, bias=False)
    self.model.classifier[-1]=nn.Linear(self.model.classifier[-1].in_features, 10)

  def forward(self, x):
    return self.model(x)

class SDSS_mod(nn.Module):
    def __init__(self, input_dim=18):
        super(SDSS_mod, self).__init__()
        self.layer1 = nn.Linear(input_dim, 20)
        self.layer2 = nn.Linear(20, 50)
        self.layer3 = nn.Linear(50, 3)
        
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.softmax(self.layer3(x), dim=1)
        return x

def data_():
    dtr=''
    dte=''
    if par['dataset']=='MNIST':
        dtr=datasets.MNIST(download=True, train=True, root=".").data.float()
        if par['model']=='resnet50':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='vgg16':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='mobilenet_v2':
            trans=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        dtr=datasets.MNIST(download=True, train=True, root=".",transform=trans)
        dte=datasets.MNIST(download=True, train=False, root=".",transform=trans)
    
    elif par['dataset']=='FMNIST':
        dtr=datasets.FashionMNIST(download=True, train=True, root=".").data.float()
        if par['model']=='resnet50':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='vgg16':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='mobilenet_v2':
            trans=transforms.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor(),transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        dtr=datasets.FashionMNIST(download=True, train=True, root=".",transform=trans)
        dte=datasets.FashionMNIST(download=True, train=False, root=".",transform=trans)

    elif par['dataset']=='CIFAR10':
        dtr=datasets.CIFAR10(download=True, train=True, root=".").data
        if par['model']=='resnet50':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='vgg16':
            trans=transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        dtr=datasets.CIFAR10(download=True, train=True, root=".",transform=trans)
        dte=datasets.CIFAR10(download=True, train=False, root=".",transform=trans)

    elif par['dataset']=='SDSS':

        # with zipfile.ZipFile('SDSS_data.zip', 'r') as zip_ref:
        #     zip_ref.extractall('SDSS_data')

        l=os.listdir('SDSS_data')
        df=pd.read_csv('SDSS_data/'+l[0])
        df.drop(columns=['objid','specobjid'],inplace=True)

        for i in l[1:]:
            df2=pd.read_csv('SDSS_data/'+i)
            df2.drop(columns=['objid','specobjid'],inplace=True)
            df=pd.concat([df,df2],ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        dtr=df.iloc[:int(df.shape[0]*0.7)]
        dte=df.iloc[int(df.shape[0]*0.7):]
        ytr=torch.tensor(np.array(pd.factorize(dtr['class'])[0]))
        yte=torch.tensor(np.array(pd.factorize(dte['class'])[0]))
        xtr=torch.tensor(dtr.drop(columns=['class']).values)
        xte=torch.tensor(dte.drop(columns=['class']).values)
        dtr=TensorDataset(xtr,ytr)
        dte=TensorDataset(xte,yte)

    return (dtr,dte)


class Metric(object):
    def __init__(self):
        self.sum=torch.tensor(0.)
        self.n=torch.tensor(0.)

    def update(self, val):
        self.sum+=hvd.allreduce(val.detach().cpu())
        self.n+=1

    @property
    def avg(self):
        return self.sum/self.n


def train(epoch):

    model.train()
    tr_sampler.set_epoch(epoch)
    tr_loss=Metric()
    tr_acc=Metric()

    with tqdm(total=len(train_loader), desc='Train Epoch #{}'.format(epoch + 1),disable=(hvd.rank()!=0)) as t:
        for bid, (x, y) in enumerate(train_loader):

            for param_group in optimizer.param_groups:
                param_group['lr']=0.0125* hvd.size()*0.01

            if par['cuda']:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            
            for i in range(0, len(x), par['batch_size']):

                (xbt,ybt)=(x[i:i+par['batch_size']], y[i:i+par['batch_size']])
                if par['dataset']=='SDSS':
                    pred=model(xbt.float())
                else:
                    pred=model(xbt)
                tr_acc.update(accuracy(pred, ybt))
                loss=nn.functional.cross_entropy(pred, ybt)
                tr_loss.update(loss)
                
                loss.div_(math.ceil(float(len(x))/par['batch_size']))
                loss.backward()
                
            optimizer.step()
            t.set_postfix({'accuracy': 100.0*tr_acc.avg.item()})
            t.update(1)


def test(epoch):
    model.eval()
    te_loss=Metric()
    te_acc=Metric()

    with tqdm(total=len(te_loader), desc='Test Epoch  #{}'.format(epoch + 1),disable=(hvd.rank()!=0)) as t:

        with torch.no_grad():
            for x, y in te_loader:
                if par['cuda']:
                    x, y = x.cuda(), y.cuda()

                if par['dataset']=='SDSS':
                    pred=model(x.float())
                else:
                    pred=model(x)
                te_loss.update(nn.functional.cross_entropy(pred, y))
                te_acc.update(accuracy(pred, y))
                t.set_postfix({'accuracy': 100.0*te_acc.avg.item()})
                t.update(1)


def accuracy(output, target):
    pred=output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


if __name__ == '__main__':

    hvd.init()
    torch.manual_seed(42)
    kwargs={}

    if par['cuda']:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(42)
        kwargs={'num_workers': par['nproc'], 'pin_memory': True}

    cudnn.benchmark=True

    torch.set_num_threads(par['nproc'])

    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context']='forkserver'

    (dtr,dte)=data_()

    tr_sampler=distributed.DistributedSampler(dtr, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader=DataLoader(dtr,batch_size=par['batch_size'],sampler=tr_sampler, **kwargs)

    te_sampler=distributed.DistributedSampler(dte, num_replicas=hvd.size(), rank=hvd.rank())
    te_loader=DataLoader(dte,batch_size=par['batch_size'],sampler=te_sampler, **kwargs)

    model=resnet50_mod()
    if par['model']=='vgg16':
        model=vgg16_mod()
    elif par['model']=='mobilenet_v2':
        model=models.mobilenet.mobilenet_v2(pretrained=True)
        model.classifier[1]=nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    elif par['dataset']=='SDSS':
        model=SDSS_mod()

    if par['cuda']:
        model.cuda()

    optimizer=optim.SGD(model.parameters(),lr=0.1)

    optimizer=hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(),compression=hvd.Compression.none,
                                         backward_passes_per_step=1,op=hvd.Average,gradient_predivide_factor=1)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epcs in range(par['epochs']):
        tt=int((time.time_ns())/1000)
        train(epcs)
        if(hvd.rank()==0):
            print(epcs,int((time.time_ns())/1000)-tt)
        test(epcs)

# %tb