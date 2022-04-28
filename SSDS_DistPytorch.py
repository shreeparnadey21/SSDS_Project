from tqdm import tqdm
import torch
from torch import nn, optim
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, distributed
from torchvision import datasets, transforms, models
import os

par={'nproc':2,'batch_size':32, 'epochs':1, 'cuda':torch.cuda.is_available(), 'model':'resnet50', 'dataset':'MNIST'}

def data_():
    dtr=datasets.MNIST(download=True, train=True, root=".").data.float()
    if par['model']=='resnet50':
        trans=transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(), 
            transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
    elif par['model']=='vgg16':
        trans=transforms.Compose([ transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)),transforms.ToTensor(), 
            transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
    dtr=datasets.MNIST(download=True, train=True, root=".",transform=trans)
    dte=datasets.MNIST(download=True, train=False, root=".",transform=trans)

    if par['dataset']=='FMNIST':
        dtr=datasets.FashionMNIST(download=True, train=True, root=".").data.float()
        if par['model']=='resnet50':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='vgg16':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        dtr=datasets.FashionMNIST(download=True, train=True, root=".",transform=trans)
        dte=datasets.FashionMNIST(download=True, train=False, root=".",transform=trans)

    elif par['dataset']=='CIFAR10':
        dtr=datasets.CIFAR10(download=True, train=True, root=".").data
        if par['model']=='resnet50':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=1), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        elif par['model']=='vgg16':
            trans=transforms.Compose([ transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)),transforms.ToTensor(), 
                transforms.Normalize((dtr.mean()/255,), (dtr.std()/255,))])
        dtr=datasets.CIFAR10(download=True, train=True, root=".",transform=trans)
        dte=datasets.CIFAR10(download=True, train=False, root=".",transform=trans)

    elif par['dataset']=='SDSS':

        with zipfile.ZipFile('SDSS_data.zip', 'r') as zip_ref:
            zip_ref.extractall('SDSS_data')

        l=os.listdir('SDSS_data')
        df=pd.read_csv('SDSS_data/'+l[0])
        df.drop(columns=['objid','specobjid'],inplace=True)

        for i in l[1:]:
            df2=pd.read_csv('SDSS_data/'+i)
            df2.drop(columns=['objid','specobjid'],inplace=True)
            df=pd.concat([df,df2],ignore_index=True)

        dtr=df.iloc[:int(df.shape[0]*0.7)]
        dte=df.iloc[int(df.shape[0]*0.7):]
        ytr=torch.tensor(np.array(pd.factorize(dtr['class'])[0]))
        yte=torch.tensor(np.array(pd.factorize(dte['class'])[0]))
        xtr=torch.tensor(dtr.drop(columns=['class']).values)
        xte=torch.tensor(dte.drop(columns=['class']).values)
        dtr=TensorDataset(xtr,ytr)
        dte=TensorDataset(xte,yte)

    return (dtr,dte)

class resnet50_mod(nn.Module):
  def __init__(self, in_channels=1):
    super(resnet50_mod, self).__init__()

    self.model = models.resnet50(pretrained=True)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.fc = nn.Linear(self.model.fc.in_features, 10)

  def forward(self, x):
    return self.model(x)

if __name__ == '__main__':

    # for rank in range(par['nproc']):

    rank=os.environ['LOCAL_RANK']

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,init_method='env://')
    device=torch.device(f'cuda:{rank}')

    (dtr,dte)=data_()

    tr_sampler=distributed.DistributedSampler(dtr, num_replicas=par['nproc'], rank=rank, shuffle=True, seed=42)
    train_loader = torch.utils.data.DataLoader(dtr,batch_size=par['batch_size'],sampler=tr_sampler)
    
    te_sampler=distributed.DistributedSampler(dte, num_replicas=par['nproc'], rank=rank, shuffle=True, seed=42)
    te_loader = torch.utils.data.DataLoader(dte,batch_size=par['batch_size'],sampler=dte_sampler)


    model=resnet50_mod()
    if par['model']=='vgg16':
        model = vgg16_mod()
    elif par['dataset']=='SDSS':
        model = nn.Sequential(nn.Linear(18, 30),nn.ReLU(),nn.Linear(30,3),nn.ReLU(),nn.Flatten(3, 1))

    model=DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    for i in range(epochs):
        
        model.train()
        tr_loader.sampler.set_epoch(i)

        epoch_loss = 0
        # train the model for one epoch
        pbar = tqdm(tr_loader)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x = x.view(x.shape[0], -1)
            optimizer.zero_grad()
            y_hat = model(x)
            batch_loss = loss(y_hat, y)
            batch_loss.backward()
            optimizer.step()
            batch_loss_scalar = batch_loss.item()
            epoch_loss += batch_loss_scalar / x.shape[0]
            pbar.set_description(f'training batch_loss={batch_loss_scalar:.4f}')

        with torch.no_grad():
            model.eval()
            te_loss = 0
            pbar = tqdm(te_loader)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                y_hat = model(x)
                batch_loss = loss(y_hat, y)
                batch_loss_scalar = batch_loss.item()

                te_loss += batch_loss_scalar / x.shape[0]
                pbar.set_description(f'validation batch_loss={batch_loss_scalar:.4f}')

        print(f"Epoch={i}, train_loss={epoch_loss:.4f}, te_loss={te_loss:.4f}")