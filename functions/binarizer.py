# imports
from torch.autograd import Function


"""
Binarize

    forward and backward methods for STE binarization function
    
    ref: 
        https://arxiv.org/pdf/1511.06085.pdf
        https://github.com/1zb/pytorch-image-comp-rnn/blob/master/functions/sign.py
    
"""


class Binarize(Function):

    @staticmethod
    def forward(ctx, input, is_train=True):
        # [-1,1] to be converted to {-1,1} by uniform noise
        x = input.clone()

        if is_train:
            # during training
            p = input.new(input.size()).uniform_()
            x[(1-input)/2 <= p] = 1
            x[(1-input)/2 > p] = -1

        else:
            # inference
            x = x.sign()

        return x
    
    @staticmethod
    def backward(ctx, grad_out):
        # backward: (grad_out), output gradient is passed through unchanged (STE)
        grad_in = grad_out.clone()
        return grad_in, None
