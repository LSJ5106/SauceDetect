import numpy as np
import torch
from torch import nn
from torch.nn import init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1)).to(device)
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1)).to(device)
        self.softmax_channel=nn.Softmax(1).to(device)
        self.softmax_spatial=nn.Softmax(-1).to(device)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1)).to(device)
        self.ln=nn.LayerNorm(channel).to(device)
        self.sigmoid=nn.Sigmoid().to(device)
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1)).to(device)
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1)).to(device)
        self.agp=nn.AdaptiveAvgPool2d((1,1)).to(device)

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out

def module_4(x):
    psa = SequentialPolarizedSelfAttention(channel=512)
    return psa(x)

def module_3(x):
    psa = SequentialPolarizedSelfAttention(channel=256)
    return psa(x)

def module_2(x):
    psa = SequentialPolarizedSelfAttention(channel=128)
    return psa(x)

def module_1(x):
    psa = SequentialPolarizedSelfAttention(channel=64)
    return psa(x)

# 参数量计算函数
def calculate_parameters(module):
    return sum(p.numel() for p in module.parameters())

if __name__ == '__main__':
    input=torch.randn(2,512,64,64).cuda()
    psa = SequentialPolarizedSelfAttention(channel=512)
    output=psa(input)
    print(output.shape)

    total_parameters = 0
    total_parameters += calculate_parameters(SequentialPolarizedSelfAttention(channel=512))
    total_parameters += calculate_parameters(SequentialPolarizedSelfAttention(channel=256))
    total_parameters += calculate_parameters(SequentialPolarizedSelfAttention(channel=128))
    total_parameters += calculate_parameters(SequentialPolarizedSelfAttention(channel=64))

    print("Number of parameter: %.4fM" % (total_parameters / 1e6))
