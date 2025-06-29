import torch 
import torch.nn as nn

architecture_config =[
    # Tuple: (kernel_size, filters, stride, padding)
    (7,64, 2, 3), 
    "M", # Max pooling layer
    (3, 192, 1, 1),
    "M",
    (1,128,1,1),
    (3,256,1,1),
    (1,256,1,0),
    (3,512,1,1),
    "M",
    # List : tuples and the last integer represents no. of repeats
    [(1,256,1,0), (3,512,1,1),4],
    (1,512,1,0),
    (3,1024,1,1),
    "M",
    [(1,512, 1,0), (3,1024,1,1),2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1),
    ]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        '''
        self.in_channels= in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        '''
        self.conv= nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) #creates an object
        self.batchnorm= nn.BatchNorm2d(out_channels)
        self.leakyrelu= nn.LeakyReLU(0.1)
    def forward(self,x):
        x= self.conv(x) #runs as a fn
        x= self.batchnorm(x)
        x= self.leakyrelu(x)
        return x

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.in_channels= in_channels
        self.architecture = architecture_config
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
        
    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x)== tuple:
                layers += [
                    CNNBlock(
                    in_channels, 
                    out_channels= x[1],
                    kernel_size=x[0], 
                    stride= x[2] ,
                    padding = x[3])
                ]
                in_channels=x[1]
            elif type(x) == str:
                layers += [
                    nn.MaxPool2d(
                    kernel_size = 2,
                    stride=2
                )
                    ]
            elif type(x) == list:
                conv1 = x[0] #tuple
                conv2 = x[1] # tuple
                num_repeats = x[2] #integer
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            out_channels= conv1[1],
                            kernel_size= conv1[0],
                            stride = conv1[2],
                            padding = conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            in_channels= conv1[1],
                            out_channels= conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3],
                        )
                    ]
                    in_channels=conv2[1]
        return nn.Sequential(*layers)
    
    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496), #original paper this would be 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5)), #(S, S, 30)
        )
        
def test(split_size =7, num_boxes = 2, num_classes = 20):
    model = Yolov1(split_size=split_size, num_boxes=num_boxes, num_classes=num_classes)
    x=torch.randn((1,3,448,448))
    print(model(x).shape)
        
test()