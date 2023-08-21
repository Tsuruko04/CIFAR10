import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn,optim
import model

device = torch.device('cuda:0')

# Load data
dataset = torchvision.datasets.CIFAR10('dataset',train=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,64)
testset = torchvision.datasets.CIFAR10('dataset',train=False,transform=torchvision.transforms.ToTensor())
testloader = DataLoader(testset,64)

# Load model
network = model.Network().cuda(device)
network.load_state_dict(torch.load('network.pth'))

# Loss function
loss_fn = nn.CrossEntropyLoss() .cuda(device)   

# Optimizer
opt = optim.SGD(network.parameters(),lr=0.01)

# Length
train_len = len(dataset)
test_len = len(testset)
# Train
for epoch in range(10):
    running_loss = 0.0
    print('---epoch {} starts---'.format(epoch+1))
    for data in dataloader:
        opt.zero_grad()
        imgs,tgts = data
        imgs = imgs.cuda()
        tgts = tgts.cuda()
        output = network(imgs)
        loss=loss_fn(output,tgts)
        loss.backward()
        opt.step()
        running_loss += loss
    print('total running loss: {}'.format(running_loss.item()))
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for data in testloader:
            imgs,tgts = data
            imgs,tgts=imgs.cuda(),tgts.cuda()
            output = network(imgs)
            loss = loss_fn(output,tgts)
            test_loss += loss
            accuracy = (output.argmax(1)==tgts).sum()
            test_accuracy+=accuracy
    print('total test loss: {}'.format(test_loss.item()))
    print('total test accuracy: {}'.format(test_accuracy/test_len))
    print('---epoch {} finishes---'.format(epoch+1))
torch.save(network.state_dict(),'network.pth')