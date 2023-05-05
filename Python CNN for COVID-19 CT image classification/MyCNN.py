from numpy import vstack
from numpy import argmax
import torch
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d
from torch.nn import Dropout
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
##数据读取
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data
class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=True):
        self.root_dir = root_dir
        self.txt_path = [txt_COVID,txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
            img=image
            label=int(self.img_list[idx][1])
        return img,label
##MyCNN
class MyCNN(Module):
    def __init__(self,n_channels):
        super(MyCNN, self).__init__()
        #conv1
        self.conv1=Conv2d(n_channels, 16, 3)
        kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        #conv2
        self.conv2 = Conv2d(16, 32, 3)
        kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        #pool1
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        ##BN1
        self.bn1=BatchNorm2d(32)
        #conv3
        self.conv3 = Conv2d(32, 32, 3)
        kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # conv4
        self.conv4 = Conv2d(32, 64, 3)
        kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        #pool2
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))
        # conv5
        self.conv5 = Conv2d(64, 64, 3)
        kaiming_uniform_(self.conv5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        # conv6
        self.conv6 = Conv2d(64, 64, 3)
        kaiming_uniform_(self.conv6.weight, nonlinearity='relu')
        self.act6 = ReLU()
        #pool3
        self.pool3 = MaxPool2d((2, 2), stride=(2, 2))
        ##BN2
        self.bn2=BatchNorm2d(64)
        #fc1
        self.fc1 = Linear(24 * 24 * 64, 10000)
        kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        self.act7 = ReLU()
        #Dropout
        self.dropout=Dropout(dropoutrate)
        self.actdr=ReLU()
        #BN3
        self.bn3=BatchNorm1d(10000)
        #fc2
        self.fc2 = Linear(10000,1000)
        kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        self.act8 = ReLU()
        #BN4
        self.bn4=BatchNorm1d(1000)
        #fc3
        self.fc3 = Linear(1000,500)
        kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
        self.act9 = ReLU()
        #BN5
        self.bn5=BatchNorm1d(500)
        # output layer
        self.fc4 = Linear(500, 2)
        xavier_uniform_(self.fc4.weight)
        self.act10 = Softmax(dim=1)
    def forward(self, Out):
        #conv1
        Out = self.conv1(Out)
        Out = self.act1(Out)
        #conv2
        Out=self.conv2(Out)
        Out=self.act2(Out)
        #pool1
        Out = self.pool1(Out)
        #BN1
        Out=self.bn1(Out)
        #conv3
        Out = self.conv3(Out)
        Out = self.act3(Out)
        #conv4
        Out=self.conv4(Out)
        Out=self.act4(Out)
        #pool2
        Out = self.pool2(Out)
        # conv5
        Out = self.conv5(Out)
        Out = self.act5(Out)
        # conv6
        Out = self.conv6(Out)
        Out = self.act6(Out)
        #pool3
        Out=self.pool3(Out)
        #BN2
        Out=self.bn2(Out)
        # flatten
        Out = torch.flatten(Out, 1)
        #fc1
        Out=self.fc1(Out)
        Out = self.act7(Out)
        #Dropout
        Out=self.dropout(Out)
        Out=self.actdr(Out)
        #BN3
        Out=self.bn3(Out)
        # fc2
        Out = self.fc2(Out)
        Out = self.act8(Out)
        #BN4
        Out = self.bn4(Out)
        # fc3
        Out = self.fc3(Out)
        Out = self.act9(Out)
        #BN5
        Out = self.bn5(Out)
        # fc4output
        Out = self.fc4(Out)
        Out = self.act10(Out)
        return Out
##train
def train_model(train_dl, model):
    criterion = CrossEntropyLoss()
    # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=0.0001)
    for epoch in range(epochnum):
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            print(loss)
            loss.backward()
            optimizer.step()
#evaluate
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        yhat = argmax(yhat, axis=1)
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc
if __name__ == '__main__':
    batchsize=100
    dropoutrate=0.5
    epochnum=50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(mean=[0.5], std=[0.25])
    train_transformer = transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
    val_transformer = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
    trainset = CovidCTDataset(root_dir='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Images-processed',
                              txt_COVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\COVID\\trainCT_COVID.txt',
                              txt_NonCOVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\NonCOVID\\trainCT_NonCOVID.txt',
                              transform= train_transformer)
    valset = CovidCTDataset(root_dir='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Images-processed',
                              txt_COVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\COVID\\valCT_COVID.txt',
                              txt_NonCOVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\NonCOVID\\valCT_NonCOVID.txt',
                              transform= val_transformer)
    testset = CovidCTDataset(root_dir='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Images-processed',
                              txt_COVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\COVID\\testCT_COVID.txt',
                              txt_NonCOVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\NonCOVID\\testCT_NonCOVID.txt',
                              transform= val_transformer)
    '''
    trainset = CovidCTDataset(root_dir='D:\\coviddata\\COVID-CT\\COVID-CT-master\\image',
                              txt_COVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\COVID\\trainCT_COVID.txt',
                              txt_NonCOVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\NonCOVID\\trainCT_NonCOVID.txt',
                              transform= None)
    valset = CovidCTDataset(root_dir='D:\\coviddata\\COVID-CT\\COVID-CT-master\\image',
                              txt_COVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\COVID\\valCT_COVID.txt',
                              txt_NonCOVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\NonCOVID\\valCT_NonCOVID.txt',
                              transform= None)
    testset = CovidCTDataset(root_dir='D:\\coviddata\\COVID-CT\\COVID-CT-master\\image',
                              txt_COVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\COVID\\testCT_COVID.txt',
                              txt_NonCOVID='D:\\coviddata\\COVID-CT\\COVID-CT-master\\Data-split\\NonCOVID\\testCT_NonCOVID.txt',
                              transform= None)
    '''
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, drop_last=False, shuffle=False)
    model = MyCNN(1)
    train_model(train_loader+test_loader, model)
    acc = evaluate_model(val_loader, model)
    print('Accuracy: %.3f' % acc)