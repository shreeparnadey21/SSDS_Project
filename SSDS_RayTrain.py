import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import ray.train as train
from ray.train.trainer import Trainer
import time

par={'nproc':2,'batch_size':32, 'epochs':1, 'cuda':torch.cuda.is_available(), 'model':'resnet50', 'dataset':'MNIST'}

class resnet50_mod(nn.Module):
  def __init__(self, in_channels=1):
    super(resnet50_mod, self).__init__()

    self.model = models.resnet50(pretrained=True)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.fc = nn.Linear(self.model.fc.in_features, 10)

  def forward(self, x):
    return self.model(x)


class vgg16_mod(nn.Module):
  def __init__(self, in_channels=1):
    super(vgg16_mod, self).__init__()

    self.model = models.vgg16(pretrained=True)
    self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, bias=False)
    self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, 10)

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
        df = df.sample(frac=1).reset_index(drop=True)
        dtr=df.iloc[:int(df.shape[0]*0.7)]
        dte=df.iloc[int(df.shape[0]*0.7):]
        ytr=torch.tensor(np.array(pd.factorize(dtr['class'])[0]))
        yte=torch.tensor(np.array(pd.factorize(dte['class'])[0]))
        xtr=torch.tensor(dtr.drop(columns=['class']).values)
        xte=torch.tensor(dte.drop(columns=['class']).values)
        dtr=TensorDataset(xtr,ytr)
        dte=TensorDataset(xte,yte)

    return (dtr,dte)

def train_(dataloader, model, optimizer):
    
    model.train()
    loss_fn=nn.CrossEntropyLoss()

    for bid, (x, y) in enumerate(dataloader):
        if par['dataset']=='SDSS':
            pred=model(x.float())
        else:
            pred=model(x)
        loss=loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_(dataloader, model):
    
    model.eval()
    loss_fn=nn.CrossEntropyLoss()
    acc=0

    with torch.no_grad():
        for x, y in dataloader:
            pred=model(x)
            if par['dataset']=='SDSS':
                pred=model(x.float())
            else:
                pred=model(x)
            acc+=(pred.argmax(1)==y).type(torch.float).sum().item()

    acc/=(len(dataloader.dataset)//train.world_size())

    print('Test Accuracy:', acc)

def train_func():

    (dtr,dte)=data_()
    tr_loader=DataLoader(dtr,batch_size=par['batch_size'])
    te_loader=DataLoader(dte,batch_size=par['batch_size'])
    train_dataloader = train.torch.prepare_data_loader(tr_loader)
    test_dataloader = train.torch.prepare_data_loader(te_loader)
    
    model=resnet50_mod()
    if par['model']=='vgg16':
        model=vgg16_mod()
    elif par['model']=='mobilenet_v2':
        model=models.mobilenet.mobilenet_v2(pretrained=True)
        model.classifier[1]=nn.Linear(in_features=model.classifier[1].in_features, out_features=10)
    elif par['dataset']=='SDSS':
        model=SDSS_mod()

    model=train.torch.prepare_model(model)

    optimizer=optim.SGD(model.parameters(),lr=0.1)

    for _ in range(par['epochs']):
        tt=(time.time())
        train_(train_dataloader, model, optimizer)
        print(epcs,(time.time()-tt)/60)
        test_(test_dataloader, model)

if __name__ == "__main__":
    trainer = Trainer(backend="torch", num_workers=par['nproc'], use_gpu=True)
    trainer.start()
    trainer.run(train_func=train_func)
    trainer.shutdown()