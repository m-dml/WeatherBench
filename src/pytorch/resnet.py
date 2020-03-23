import torch
from torchvision.models.resnet import ResNet
from .layers import PeriodicConv2D


class FCNResNet(ResNet):
    """ Simple fully convolutional ResNet with variable number of blocks and layers
    Based on torchvision ResNet implementation
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L124
    """
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, in_channels=1, out_channels=1, 
                 nfilters=[64, 64, 128, 256, 512], kernel_size=3):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.inplanes = nfilters[0]
        self.conv1 = torch.nn.Conv2d(in_channels, nfilters[0], kernel_size=kernel_size, 
                                     stride=1, padding=(kernel_size-1)//2, bias=True)
        self.bn1 = norm_layer(nfilters[0])
        self.relu = torch.nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, nfilters[1], layers[0])
        if len(layers) > 2:
            self.layer2 = self._make_layer(block, nfilters[2], layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
        if len(layers) > 3:
            self.layer3 = self._make_layer(block, nfilters[3], layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
        if len(layers) > 4:
            self.layer4 = self._make_layer(block, nfilters[4], layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])

        self.final = torch.nn.Conv2d(in_channels=4*nfilters[len(layers)-1], # why 4x?
                                     out_channels=out_channels, 
                                     kernel_size=(1,1), 
                                     stride=1) 

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
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

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        # initial conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

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
