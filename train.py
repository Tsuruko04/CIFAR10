import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn,optim
import model
# Load data
dataset = torchvision.datasets.CIFAR10('dataset',train=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,64)
testset = torchvision.datasets.CIFAR10('dataset',train=False,transform=torchvision.transforms.ToTensor())
testloader = DataLoader(testset,64)

# Load model
network = model.Network().load_state_dict()

# Loss function
loss_fn = nn.CrossEntropyLoss()    

# Optimizer
opt = optim.SGD(network.parameters(),lr=0.01)

# Train
for epoch in range(10):
    running_loss = 0.0
    print('---epoch {} starts---'.format(epoch))
    for data in dataloader:
        opt.zero_grad()
        imgs,tgts = data
        output = network(imgs)
        loss=loss_fn(output,tgts)
        loss.backward()
        opt.step()
        running_loss += loss
    print('total running loss: {}'.format(running_loss.item()))
    test_loss = 0.0
    with torch.no_grad():
        for data in testloader:
            imgs,tgts = data
            output = network(imgs)
            loss = loss_fn(output,tgts)
            test_loss += loss
    print('total test loss: {}'.format(test_loss.item()))
    print('---epoch {} finishes---'.format(epoch))
torch.save(network.state_dict(),'network.pth')