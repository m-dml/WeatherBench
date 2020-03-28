import torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import Bottleneck, conv1x1
from .layers import PeriodicConv2D



def circConv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, kernel_size=3, padding_mode='circular'):
    """3x3 convolution with padding"""
    return PeriodicConv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size-1, groups=groups, bias=True, dilation=dilation,
                     padding_mode=padding_mode)

class CircBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                 base_width=64, dilation=1, norm_layer=None, 
                 kernel_size=3, padding_mode='circular', dropout_rate=0):
        super(CircBlock, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        if groups != 1:
            raise ValueError('CircBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = circConv3x3(inplanes, planes, stride, kernel_size=kernel_size, padding_mode=padding_mode)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else torch.nn.Identity()
        self.conv2 = circConv3x3(planes, planes, kernel_size=kernel_size, padding_mode=padding_mode)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out)
        out = self.dropout(out) # 2nd application of Dropout with own call to torch.functional.Dropout
            
        out += identity

        return out


class FCNResNet(ResNet):
    """ Simple fully convolutional ResNet with variable number of blocks and layers
    Based on torchvision ResNet implementation
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L124
    """
    def __init__(self, block, zero_init_residual=False,
                 groups=1, width_per_group=64,
                 norm_layer=None, in_channels=1, out_channels=1, 
                 filters=[64, 128, 128], kernel_sizes=[3,3,3],
                 padding_mode='zeros',
                 dropout_rate=0):
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group

        self.inplanes = filters[0]
        self.conv1 = PeriodicConv2D(in_channels, filters[0], kernel_size=kernel_sizes[0], 
                                    stride=1,  padding=kernel_sizes[0]-1, bias=True,
                                    padding_mode=padding_mode)
        self.bn1 = norm_layer(filters[0])
        self.relu = torch.nn.ReLU(inplace=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate) if dropout_rate > 0 else torch.nn.Identity()

        self.layer1 = self._make_layer(block, filters[1:], kernel_sizes[1:], dr=dropout_rate)
        """
        if len(layers) > 2:
            self.layer2 = self._make_layer(block, nfilters[2], layers[1], dr=dropout_rate)
        if len(layers) > 3:
            self.layer3 = self._make_layer(block, nfilters[3], layers[2], dr=dropout_rate)
        if len(layers) > 4:
            self.layer4 = self._make_layer(block, nfilters[4], layers[3], dr=dropout_rate)
        """
        expansion = 4 if block is Bottleneck else 1
        self.final = torch.nn.Conv2d(in_channels=expansion*filters[-1],
                                     out_channels=out_channels, 
                                     kernel_size=(1,1), 
                                     stride=1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d): # also covers CircularConv2D
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, kernel_sizes, stride=1, dilation=1, dr=0):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes[0] * block.expansion:
            downsample = torch.nn.Sequential(
                conv1x1(self.inplanes, planes[0] * block.expansion, stride),
                norm_layer(planes[0] * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes[0], stride, downsample, self.groups, self.base_width,
                            dilation, norm_layer, kernel_size=kernel_sizes[0], dropout_rate=dr))
        self.inplanes = planes[0] * block.expansion
        for p, k in zip(planes[1:], kernel_sizes[1:]):
            layers.append(block(self.inplanes, p, groups=self.groups, kernel_size=k,
                                base_width=self.base_width, dilation=dilation,
                                norm_layer=norm_layer, dropout_rate=dr))

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        # initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # actual ResNet layers (multiple ResNet blocks per layer)
        x = self.layer1(x)
        if hasattr(self, 'layer2'):
            x = self.layer2(x)
        if hasattr(self, 'layer3'):
            x = self.layer3(x)
        if hasattr(self, 'layer4'):
            x = self.layer4(x)

        # final 1x1 convolution
        x = self.final(x)
        
        return x
