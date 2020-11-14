import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

from time import time as t
from tqdm import tqdm
from bindsnet.datasets import MNIST,AD
from  bindsnet.datasets import ad_dataset
import  random

from bindsnet.encoding import PoissonEncoder,BernoulliEncoder,RepeatEncoder
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Conv2dConnection, Connection
import  torch.utils.data as data
import  numpy as np
import nibabel as nib
import torch.nn.functional as F
from bindsnet.utils import get_square_weights
from  bindsnet.datasets.AD_Dataset import Dataset_Import
from  bindsnet.datasets import AD_Constants
from bindsnet.datasets.torchvision_wrapper2 import  create_torchvision_dataset_wrapper2
import torch.optim as optim
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import  pandas as pd
import torchvision
from torchsummary import summary
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
torchvision.models.resnet18()
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_conv2d_weights,
    plot_voltages,
)


cfgs = {
    'A': [8, 'M', 16, 'M', 32, 32, 'M', 64, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG(nn.Module):

    def __init__(self, features, num_classes=2, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        #print("dhdhd ",x.size())
        x = x.view(-1, 64*7*7)
        #x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(cfg, batch_norm, pretrained, progress, **kwargs):

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    return model



def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('A', False, pretrained, progress, **kwargs)



class MNISTDataset(data.Dataset):

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




class CovDataset(data.Dataset):

    def __init__(self, images=None, transforms=None):
        self.X = images
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):

        data = self.X[i]
        label = data[1]
        data = np.asarray(nib.load(data[0], mmap=False).get_data()[:,:,:])

        print("My shape ",data.shape)
        data = np.resize(data,AD_Constants.img_shape_tuple)

        if self.transforms:
            data = self.transforms(data)

        return data,label






data_import =Dataset_Import()
ad_files=data_import.readNiiFiles_2(data_import.train_ad_dir)
mci_file=data_import.readNiiFiles_2(data_import.train_mci_dir)
nc_file=data_import.readNiiFiles_2(data_import.train_nc_dir)

all_files=[*ad_files,*mci_file]
random.shuffle(all_files)


ad_test=data_import.readNiiFiles_2(data_import.validation_ad_dir)
mci_test=data_import.readNiiFiles_2(data_import.validation_mci_dir)
nc_test=data_import.readNiiFiles_2(data_import.validation_nc_dir)

test_files=[*ad_test,*mci_test]


intensity=128.0
transformi=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    )


# trainloader = torch.utils.data.DataLoader(datasets, batch_size=128, shuffle=True)
#
# for data,label in trainloader:
#     print(data)
#
# exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--kernel_size", type=int, default=32)
parser.add_argument("--stride", type=int, default=128)
parser.add_argument("--n_filters", type=int, default=25)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_false")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
padding = args.padding
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

if gpu:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test





conv_size = int((464 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(shape=(1, 464, 464), traces=True)

conv_layer = DiehlAndCookNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")


if gpu:
    network.to("cuda")

# Load MNIST data.


#datasets=MNISTDataset(all_files,transformi)
train_dataset = create_torchvision_dataset_wrapper2(MNISTDataset)

train_dataset=train_dataset(RepeatEncoder(time=time, dt=dt),None,all_files,transformi,
)




test_dataset = create_torchvision_dataset_wrapper2(MNISTDataset)

test_dataset=test_dataset(RepeatEncoder(time=time, dt=dt),None,test_files,transformi,
)



spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

# Create and train logistic regression model on reservoir outputs.
class Net(Module):
    def __init__(self):
          super(Net, self).__init__()


            # Defining a 2D convolution layer
          self.conv1=  Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
          self.batch1=  BatchNorm2d(4)
          self.pool1=  MaxPool2d(kernel_size=2, stride=2)
            # Defining another 2D convolution layer
          self.conv2=  Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
          self.batch2=  BatchNorm2d(8)

          self.pool2= MaxPool2d(kernel_size=2, stride=2)

          self.fcn = Linear(8*12*53824, 10)
          self.fcn2=Linear(10, 2)


    # Defining the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x= F.relu(self.batch1(x))
        x=self.pool1(x)
        x=self.conv2(x)
        x=F.relu(self.batch2(x))
        x=self.pool2(x)
        #print("sha ",x.size())  #4*12*100
        x = x.view(-1,8*12*53824)
        x=self.fcn(x)
        x=F.dropout(x,0.5)
        x=self.fcn2(x)

        return x



# Create and train logistic regression model on reservoir outputs.
class Net2(Module):
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

          self.fcn = Linear(4*12*53824, 10)
          self.fcn2=Linear(10, 2)


    # Defining the forward pass
    def forward(self, x):
        x = self.conv1(x)
        x= F.relu(self.batch1(x))
        x=self.pool1(x)
        x=self.conv2(x)
        x=F.relu(self.batch2(x))
        x=self.pool2(x)
        #print("sha ",x.size())
        x = x.view(-1,4*12*53824)
        x=self.fcn(x)
        x=F.dropout(x,0.5)
        x=self.fcn2(x)

        return x



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#model=vgg16().to(device)


#print(model)
#summary(model, (1, 464, 464))



model = Net()






criterion =nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
optimizer2 = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

if torch.cuda.is_available():
    model = model.to(device)
    criterion = criterion.to(device)


for epoch in tqdm(range(n_epochs)):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    batch_size=4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=gpu
    )

    if epoch == n_epochs-1:

        #network.train(mode=False)
        print("Classification Training")

        batch_size = 4
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=gpu
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=gpu
        )


        avg_loss = 0
        correct = 0
        total = 0
        train_accs = []
        train_losses = []
        eval_list = []
        df = pd.DataFrame([], columns=['Accuracy', "Loss", "Test Acc", "Test Loss"])

        for epoch in range(200):

            for step, batch in enumerate(train_dataloader):
                # Get next input sample.


                size_use = batch["encoded_image"].size()[0]

                inputs = {"X": batch["encoded_image"].view(time, size_use, 1, 464, 464)}
                if gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                label = batch["label"]

                #print(label)

                # Run the network on the input.
                network.run(inputs=inputs, time=time, input_time_dim=1)

                #data_x = voltages["Y"].get("v").view(batch_size,1,time, -1)
                data_x= spikes["X"].get("s").view(size_use,1,time, -1)

                #print("sgsgs ",data_x.size())

                print("output shape ",spikes["X"].get("s").size())


                data_x = Variable(data_x)
                label = Variable(label)

                #print("spike ",data_x.size())
                if torch.cuda.is_available():
                    data_x = data_x.to(device, dtype=torch.float)
                    label = label.to(device)

                #print("size all",data_x[0].size())
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(data_x)



                #accuracy calculate
                # accuracy calculation
                softmax = torch.exp(outputs).cpu()
                prob = list(softmax.cpu().detach().numpy())
                predictions = np.argmax(prob, axis=1)
                acc = accuracy_score(label.cpu(), predictions)
                train_accs.append(acc)

                #print(outputs.data)

                loss = criterion(outputs, label)

                train_losses.append(loss.cpu().detach().numpy())
                loss.backward()
                optimizer.step()

                network.reset_state_variables()

            test_acc = []
            test_loss = []
            for step, batch in enumerate(test_dataloader):

                size_use = batch["encoded_image"].size()[0]
                inputs = {"X": batch["encoded_image"].view(time, size_use, 1, 464, 464)}
                label = batch["label"]

                #network.train(mode=False)

                network.run(inputs=inputs, time=time, input_time_dim=1)

                x_test = spikes["X"].get("s").view(size_use, 1, time, -1)


                x_test, y_test = Variable(x_test), Variable(label)
                # getting the validation set
                # converting the data into GPU format
                if torch.cuda.is_available():
                    x_test = x_test.to(device, dtype=torch.float)
                    y_test = y_test.to(device)

                with torch.no_grad():
                    output_test = model(x_test)

                #print("step ",step)
                loss_test = criterion(output_test, y_test)
                test_loss.append(loss_test.cpu().detach().numpy())

                softmax = torch.exp(output_test).cpu()
                test_prob = list(softmax.cpu().detach().numpy())
                test_predictions = np.argmax(test_prob, axis=1)
                tacc = accuracy_score(y_test.cpu(), test_predictions)
                test_acc.append(tacc)

                #network.reset_state_variables()

            eval_list.append([round(np.mean(train_accs),3),
                              round(np.mean(train_losses),3), round(np.mean(test_acc), 3), round(np.mean(test_loss), 3)])

            test_check=round(np.mean(test_acc), 3)

            if test_check>=0.7:
                PATH = './'+str(test_check)+'ad_net.pth'
                torch.save(model.state_dict(), PATH)

            data = pd.DataFrame(eval_list, columns=['Accuracy', "Loss", "Test Acc", "Test Loss"])
            data.to_csv("train_results.csv")
            print("Step {}/{} loss : {} Acc : {} Test loss : {} Test Acc : {} ".format(epoch + 1,200, round(np.mean(train_losses),3),round(np.mean(train_accs),3),
                                                                                       round(np.mean(test_loss), 3),round(np.mean(test_acc), 3)
                                                                                       ))



    else:


        print("Spike Training ",epoch,end="\n")

        for step, batch in enumerate(train_dataloader):
            # Get next input sample.
            size_use=batch["encoded_image"].size()[0]
            inputs = {"X": batch["encoded_image"].view(time, size_use, 1, 464, 464)}
            if gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            label = batch["label"]

            # Run the network on the input.
            network.run(inputs=inputs, time=time, input_time_dim=1)

            if plot:
                inpt_axes, inpt_ims = plot_input(
                    batch["encoded_image"].view(28, 28),
                    batch["encoded_image"].view(time, 784).sum(0).view(28, 28),
                    label=label,
                    axes=inpt_axes,
                    ims=inpt_ims,
                )
                spike_ims, spike_axes = plot_spikes(
                    {layer: spikes[layer].get("s").view(-1, 250) for layer in spikes},
                    axes=spike_axes,
                    ims=spike_ims,
                )
                voltage_ims, voltage_axes = plot_voltages(
                    {layer: voltages[layer].get("v").view(-1, 250) for layer in voltages},
                    ims=voltage_ims,
                    axes=voltage_axes,
                )
                weights_im = plot_weights(
                    get_square_weights(conv_conn.w, 23, 28), im=weights_im, wmin=-2, wmax=2
                )
                weights_im2 = plot_weights(recurrent_conn.w, im=weights_im2, wmin=-2, wmax=2)

                plt.pause(1e-8)
                plt.show()




        # Optionally plot various simulation information.
        # if plot:
        #     image = batch["image"].view(464, 464)
        #
        #     inpt = inputs["X"].view(time, 215296).sum(0).view(464, 464)
        #     weights1 = conv_conn.w
        #     _spikes = {
        #         "X": spikes["X"].get("s").view(time, -1),
        #         "Y": spikes["Y"].get("s").view(time, -1),
        #     }
        #     _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}
        #
        #     inpt_axes, inpt_ims = plot_input(
        #         image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
        #     )
        #     spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
        #     weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
        #     voltage_ims, voltage_axes = plot_voltages(
        #         _voltages, ims=voltage_ims, axes=voltage_axes
        #     )
        #
        #     plt.pause(1)
        #
        #     plt.show()

        network.reset_state_variables()  # Reset state variables.










# Training the Model
print("\n  Cov Training the read out")
