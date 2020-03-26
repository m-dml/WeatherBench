import numpy as np
import torch
import torchvision
from .resnet import FCNResNet, CircBlock
from .cnn import SimpleCNN
from .unet import CircUNet


def init_torch_device():
    if torch.cuda.is_available():
        print('using CUDA !')
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        print("CUDA not available")
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    return device


def named_network(model_name, n_channels):
    if model_name == 'cnnbn':

        model = SimpleCNN(filters=[64, 64, 64, 64, 2],  # last '2' for Z500, T850
                          kernels=[5, 5, 5, 5, 5],
                          channels=n_channels,
                          activation=torch.nn.functional.elu,
                          mode='circular')

        def model_forward(input):
            return model.forward(input)

    elif model_name == 'Unetbn':

        model = CircUNet(in_channels=n_channels,
                         filters=[[32], [32], [32], [32]],
                         kernels=[[5], [5], [5], [5]],
                         pooling=2,
                         activation=torch.nn.functional.elu,
                         out_channels=2,
                         mode='circular')

        def model_forward(input):
            return model.forward(input)

    elif model_name == 'tvfcnResnet50':

        k = 3

        model = torchvision.models.segmentation.fcn_resnet50(pretrained=False)

        # modify input layer (torchvision ResNet expects 3 input channels)
        model._modules['backbone']['conv1'] = torch.nn.Conv2d(
            in_channels=n_channels, out_channels=64,
            kernel_size=(k, k), stride=1, padding=(k + 1) // 2
        )
        # modify output layer (torchvision ResNet predicts 21 output channels)
        model._modules['classifier'][-1] = torch.nn.Conv2d(
            in_channels=512, out_channels=2,
            kernel_size=(k, k), stride=1, padding=(k + 1) // 2
        )

        def model_forward(input):
            return model.forward(input)['out']  # because reasons...

    elif model_name == 'simpleResnet':

        """
        from src.pytorch.resnet import FCNResNet
        from torchvision.models.resnet import Bottleneck
        model = FCNResNet(in_channels=n_channels,
                          out_channels=2,
                          block=Bottleneck, # basic ResNet block. 'Bottleneck' is 1x1 -> 3x3 -> 1x1 convs stacked  
                          replace_stride_with_dilation=[True, True, True], # assures stride=1 through all layers
                          layers=[4], # number of blocks per layer. len(layers) gives number of layers !
                          nfilters = [64, 64, 128, 256, 512], # number of filters per layer
                          kernel_size = 3 # kernel size for first conv layer
                         )
        """

        model = FCNResNet(in_channels=n_channels,
                          out_channels=2,
                          block=CircBlock,  # basic ResNet block. 'Bottleneck' is 1x1 -> 3x3 -> 1x1 convs stacked
                          # replace_stride_with_dilation=[True, True, True], # assures stride=1 through all layers
                          layers=[13],  # number of blocks per layer. len(layers) gives number of layers !
                          nfilters=[128, 128],  # number of filters per layer
                          kernel_size=7,  # kernel size for first conv layer
                          dropout_rate=0.1,
                          padding_mode='circular'
                          )

        def model_forward(input):
            return model.forward(input)

        model_name = model_name + '_' + str(2 + 2 * np.sum(layers))

    else:
        raise NotImplementedError()

    return model, model_forward