import torch
import torch.nn.functional as F


class PeriodicConv2D(torch.nn.Conv2d):
    """ Implementing 2D convolutional layer with mixed zero- and circular padding.
    Uses circular padding along last axis (W) and zero-padding on second-last axis (H)

    """
    
    def conv2d_forward(self, input, weight):
        if self.padding_mode == 'circular':
            expanded_padding_circ = ( (self.padding[0] + 1) // 2, self.padding[0] // 2, 0, 0)
            expanded_padding_zero = ( 0, 0, (self.padding[1] + 1) //2, self.padding[1] // 2 )
            return F.conv2d(F.pad(F.pad(input, expanded_padding_circ, mode='circular'), 
                                  expanded_padding_zero, mode='constant'),
                            weight, self.bias, self.stride,
                            (0,0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
