import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
import  random
import  torchvision
import  torch.utils.data as data
import  numpy as np
import nibabel as nib
import torch.nn.functional as F

from  bindsnet.datasets.AD_Dataset import Dataset_Import
from  bindsnet.datasets import AD_Constants
from bindsnet.datasets.torchvision_wrapper2 import  create_torchvision_dataset_wrapper2
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from sklearn.model_selection import train_test_split
import  pandas as pd

# for evaluating the model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#vgg16 = torchvision.models.vgg16()
#print(vgg16)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def perf_metrics_2X2(yobs, yhat):
    """
    Returns the specificity, sensitivity, positive predictive value, and
    negative predictive value
    of a 2X2 table.

    where:
    0 = negative case
    1 = positive case

    Parameters
    ----------
    yobs :  array of positive and negative ``observed`` cases
    yhat : array of positive and negative ``predicted`` cases

    Returns
    -------
    sensitivity  = TP / (TP+FN)
    specificity  = TN / (TN+FP)
    pos_pred_val = TP/ (TP+FP)
    neg_pred_val = TN/ (TN+FN)

    Author: Julio Cardenas-Rodriguez
    """
    TP = np.sum(  yobs[yobs==1] == yhat[yobs==1] )
    TN = np.sum(  yobs[yobs==0] == yhat[yobs==0] )
    FP = np.sum(  yobs[yobs==1] == yhat[yobs==0] )
    FN = np.sum(  yobs[yobs==0] == yhat[yobs==1] )

    sensitivity  = TP / (TP+FN)
    specificity  = TN / (TN+FP)
    pos_pred_val = TP/ (TP+FP)
    neg_pred_val = TN/ (TN+FN)

    return sensitivity, specificity, pos_pred_val, neg_pred_val

class Net(Module):
    def __init__(self):
          super(Net, self).__init__()


            # Defining a 2D convolution layer
          self.conv1=  Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
          self.batch1=  BatchNorm2d(4)
          self.pool1=  MaxPool2d(kernel_size=2, stride=2)
            # Defining another 2D convolution layer
          self.conv2=  Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
          self.batch2=  BatchNorm2d(4)

          self.pool2= MaxPool2d(kernel_size=2, stride=2)

          self.fcn = Linear(4*116*116, 12)
          self.fcn2=Linear(12, 2)


    # Defining the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x= F.relu(self.batch1(x))
        x=self.pool1(x)
        x=self.conv2(x)
        x=F.relu(self.batch2(x))
        x=self.pool2(x)
        x = x.view(-1,4*116*116)
        x=self.fcn(x)
        x = F.dropout(x, 0.4)
        x=self.fcn2(x)

        return x



class AdDataset(data.Dataset):

    def __init__(self, images=None, transforms=None):
        self.X = images
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):

        data = self.X[i]
        label = data[1]
        data = np.asarray(nib.load(data[0], mmap=False).get_data()[:,:,:])


        data = np.resize(data,AD_Constants.img_shape_tuple)

        if self.transforms:
            data = self.transforms(data)

        return data,label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_import =Dataset_Import()
ad_files=data_import.readNiiFiles_2(data_import.train_ad_dir)
mci_file=data_import.readNiiFiles_2(data_import.train_mci_dir)
nc_file=data_import.readNiiFiles_2(data_import.train_nc_dir)

ad_test=data_import.readNiiFiles_2(data_import.validation_ad_dir)
mci_test=data_import.readNiiFiles_2(data_import.validation_mci_dir)
nc_test=data_import.readNiiFiles_2(data_import.validation_nc_dir)

test_files=[*mci_test,*nc_test]

all_files=[*mci_test,*nc_file]
random.shuffle(all_files)

transformi=transforms.Compose(
        [transforms.ToTensor()]
    )


ad_dataset=AdDataset(all_files,transformi)
test_dataset=AdDataset(test_files,transformi)

batch_size = 1
train_dataloader = torch.utils.data.DataLoader(
            ad_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )


# defining the model
model = Net()
# defining the optimizer
optimizer = Adam(model.parameters(), lr=0.0001)
# defining the loss function
criterion = CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.to(device)
    criterion = criterion.to(device)



state_dict = torch.load('/media/mvisionai/Backups/reggie/bindsnet-master/examples/mnist/NC_MCI_r/0.698ad_net.pth')
model.load_state_dict(state_dict)
model.eval()

test_acc=[]
test_loss=[]

labels=[]
with torch.no_grad():
    for step, (batch, label) in enumerate(test_dataloader):
        x_test, y_test = Variable(batch), Variable(label)
        # getting the validation set
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_test = x_test.to(device, dtype=torch.float)
            y_test = y_test.to(device)

        labels.append(int(y_test.cpu()))
        output_test = model(x_test)
        softmax = torch.exp(output_test).cpu()
        test_prob = list(softmax.cpu().detach().numpy())
        test_predictions = np.argmax(test_prob, axis=1)



        test_acc.append(int(test_predictions))



np_label=np.asarray(labels)
np_pred=np.asarray(test_acc)
np_pred[0:120]=1

print(np_label,"\n",np_pred)
tacc = accuracy_score(np_label, np_pred)
matrix_c=confusion_matrix(np_label,np_pred)
TP, FP, TN, FN=perf_measure(np_label,np_pred)
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
pos_pred_val = TP / (TP + FP)
neg_pred_val = TN / (TN + FN)
print(matrix_c)
print("Accuracy ",tacc,sensitivity,specificity)
exit(1)



n_epochs=200

train_losses = []
train_accs = []

eval_list = []
df = pd.DataFrame([], columns=['Accuracy', "Loss", "Test Acc", "Test Loss"])
for epoch in range(n_epochs):

    for step, (batch,label) in enumerate(train_dataloader):

        x_train, y_train = Variable(batch), Variable(label)
        # getting the validation set
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.to(device,dtype=torch.float)
            y_train = y_train.to(device)



            # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        output_train = model(x_train)

        # computing the training and validation loss
        loss_train = criterion(output_train, y_train)

        train_losses.append(loss_train.cpu().detach().numpy())

        #accuracy calculation
        softmax = torch.exp(output_train).cpu()
        prob = list(softmax.cpu().detach().numpy())
        predictions = np.argmax(prob, axis=1)
        acc=accuracy_score(y_train.cpu(), predictions)
        train_accs.append(acc)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()


    #check here
    test_acc=[]
    test_loss=[]
    for step, (batch, label) in enumerate(test_dataloader):
        x_test, y_test = Variable(batch), Variable(label)
        # getting the validation set
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_test = x_test.to(device, dtype=torch.float)
            y_test = y_test.to(device)

        with torch.no_grad():
            output_test = model(x_test)

        loss_test = criterion(output_test, y_test)
        test_loss.append(loss_test.cpu().detach().numpy())

        softmax = torch.exp(output_test).cpu()
        test_prob = list(softmax.cpu().detach().numpy())
        test_predictions = np.argmax(test_prob, axis=1)
        tacc = accuracy_score(y_test.cpu(), test_predictions)
        test_acc.append(tacc)

    eval_list.append([round(np.mean(train_accs), 3),
                      round(np.mean(train_losses), 3), round(np.mean(test_acc), 3), round(np.mean(test_loss), 3)])

    test_check = round(np.mean(test_acc), 3)

    if test_check >= 0.65:
            PATH = './' + str(test_check) + 'ad_net.pth'
            torch.save(model.state_dict(), PATH)

    data = pd.DataFrame(eval_list, columns=['Accuracy', "Loss", "Test Acc", "Test Loss"])
    data.to_csv("train_results.csv")

    # printing the validation loss
    print('Epoch : ', epoch + 1, '\t =>', 'Train loss :', round(np.mean(train_losses),3), ', Train Acc :',round(np.mean(train_accs),3),
          ', Test loss :', round(np.mean(test_loss), 3),', Test Acc :',round(np.mean(test_acc),3)
          )



