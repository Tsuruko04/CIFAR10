from torch import nn
from torch.nn import Sequential,Conv2d,MaxPool2d,Flatten,Linear
class Network(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod1 = Sequential(
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
        
    def forward(self,x):
        x = self.mod1(x)
        return x
    
        