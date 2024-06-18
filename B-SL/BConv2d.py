import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ********************* 二值(+-1) ***********************
# A 对激活值进行二值化的具体实现，原理中的第一个公式
class Binary_a(Function):

    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        # *******************ste*********************
        grad_input = grad_output.clone()
        # ****************saturate_ste***************
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        '''
        #******************soft_ste*****************
        size = input.size()
        zeros = torch.zeros(size).cuda()
        grad = torch.max(zeros, 1 - torch.abs(input))
        #print(grad)
        grad_input = grad_output * grad
        '''
        return grad_input


# W 对权重进行二值化的具体实现
class Binary_w(Function):

    @staticmethod
    def forward(self, input):
        output = torch.sign(input)
        return output

    @staticmethod
    def backward(self, grad_output):
        # *******************ste*********************
        grad_input = grad_output.clone()
        return grad_input


# ********************* A(特征)量化(二值) ***********************
# 因为我们使用的网络结构不是完全的二值化，第一个卷积层是普通卷积接的ReLU激活函数，所以要判断一下
class activation_bin(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.relu = nn.ReLU(inplace=True)

    def binary(self, input):
        output = Binary_a.apply(input)
        return output

    def forward(self, input):
        if self.A == 2:
            output = self.binary(input)
            # ******************** A —— 1、0 *********************
            # a = torch.clamp(a, min=0)
        else:
            output = self.relu(input)
        return output


# ********************* W(模型参数)量化(三/二值) ***********************
def meancenter_clampConvParams(w):
    mean = w.data.mean(1, keepdim=True)
    w.data.sub(mean)  # W中心化(C方向)
    w.data.clamp(-1.0, 1.0)  # W截断
    return w


# 对激活值进行二值化
class weight_tnn_bin(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def binary(self, input):
        output = Binary_w.apply(input)
        return output

    def forward(self, input):
        # **************************************** W二值 *****************************************
        output = meancenter_clampConvParams(input)  # W中心化+截断
        # **************** channel级 - E(|W|) ****************
        E = torch.mean(torch.abs(output), (3, 2, 1), keepdim=True)
        # **************** α(缩放因子) ****************
        alpha = E
        # ************** W —— +-1 **************
        output = self.binary(output)
        # ************** W * α **************
        output = output * alpha  # 若不需要α(缩放因子)，注释掉即可
        # **************************************** W三值 *****************************************
        return output


# ********************* 量化卷积（同时量化A/W，并做卷积） ***********************
class Conv2d_Q(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            A=2,
            W=2
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        # 实例化调用A和W量化器
        self.activation_quantizer = activation_bin(A=A)
        self.weight_quantizer = weight_tnn_bin(W=W)

    def forward(self, input):
        # 量化A和W
        bin_input = self.activation_quantizer(input)
        tnn_bin_weight = self.weight_quantizer(self.weight)
        # print(bin_input)
        # print(tnn_bin_weight)
        # 用量化后的A和W做卷积
        output = F.conv2d(
            input=bin_input,
            weight=tnn_bin_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return output
