# 导入PyTorch库
import torch
import torch.nn as nn

# 定义一个多尺度融合模块
class MultiScaleFusion(nn.Module):
    def __init__(self):
        super(MultiScaleFusion, self).__init__()
        # 定义一个1x1卷积层，用于调整通道数
        self.conv1 = nn.Conv2d(64, 512, kernel_size=1).cuda()
        self.conv2 = nn.Conv2d(128, 512, kernel_size=1).cuda()
        self.conv3 = nn.Conv2d(256, 512, kernel_size=1).cuda()
        # 定义一个上采样层，用于调整空间尺寸
        self.upsample = nn.Upsample(scale_factor=2).cuda()
        # 定义一个下采样层，用于调整空间尺寸
        self.downsample = nn.MaxPool2d(kernel_size=2).cuda()

    def forward(self, x1, x2, x3, x4):
        # x1: [2,64,512,512]
        # x2: [2,128,256,256]
        # x3: [2,256,128,128]
        # x4: [2,512,64,64]
        # 调整x1的通道数为512，并下采样两次，得到[2,512,128,128]

        x1 = self.conv1(x1)
        # x1: [2, 512, 512, 512]

        x1 = self.downsample(x1)
        # x1: [2, 512, 256, 256]

        x1 = self.downsample(x1)
        # x1: [2, 512, 128, 128]

        # 调整x2的通道数为512，并下采样一次，得到[2,512,128,128]
        x2 = self.conv2(x2)
        # x2: [2, 512, 256, 256]

        x2 = self.downsample(x2)
        # x2: [2, 512, 128, 128]

        # 调整x3的通道数为512，得到[2,512,128,128]
        x3 = self.conv3(x3)
        # x3: [2, 512, 128, 128]

        # 调整x4
        x4 = self.upsample(x4)
        # x4: [2, 512, 128, 128]

        # 将x1,x2,x3,x4在通道维度上进行拼接，得到[2,2048,128,128]
        # x = torch.cat((x1,x2),dim=1)
        # x = torch.cat((x, x3), dim=1)
        # x = torch.cat((x, x4), dim=1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # x: [2, 2048, 128, 128]

        # 使用一个1x1卷积层，将通道数降低为512，得到[2,512,128.128]
        x = nn.Conv2d(2048 , 512 , kernel_size=1).cuda()(x)
        # x: [2, 512, 128, 128]


        # 下采样两次，得到[2.512.32.32]
        x = self.downsample(x)

        x = self.downsample(x)

        return x


def fusion_tensors(x1, x2, x3, x4):
    fusion = MultiScaleFusion()
    return fusion(x1 , x2 , x3 , x4)


if __name__ == '__main__':

    # 创建一个多尺度融合模块的实例
    # 创建四个随机的tensor作为输入
    x1 = torch.randn(2 , 64 , 512 , 512).cuda()
    x2 = torch.randn(2 , 128 , 256 , 256).cuda()
    x3 = torch.randn(2 , 256 , 128 , 128).cuda()
    x4 = torch.randn(2 , 512 , 64 , 64).cuda()

    output = fusion_tensors(x1,x2,x3,x4)
    # 打印输出的形状，应该是[2.512.32.32]
    print(output.shape)


