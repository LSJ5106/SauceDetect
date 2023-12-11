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
        x = nn.Conv2d(2048, 512, kernel_size=1).cuda()(x)
        # x: [2, 512, 128, 128]

        # 下采样两次，得到[2.512.32.32]
        x = self.downsample(x)

        x = self.downsample(x)

        return x


import torch
import torch.nn as nn


class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()

        # 横向连接的1x1卷积，用于调整通道数
        self.lat1 = nn.Conv2d(64, 512, kernel_size=1).cuda()
        self.lat2 = nn.Conv2d(128, 512, kernel_size=1).cuda()
        self.lat3 = nn.Conv2d(256, 512, kernel_size=1).cuda()
        self.lat4 = nn.Conv2d(512, 512, kernel_size=1).cuda()

        # 自顶向下连接的3x3卷积，用于上采样
        self.top_down1 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()
        self.top_down2 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()
        self.top_down3 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()

    def forward(self, x1, x2, x3, x4):
        # 横向连接
        lat1 = self.lat1(x1)
        lat2 = self.lat2(x2)
        lat3 = self.lat3(x3)
        lat4 = self.lat4(x4)

        # 自顶向下连接
        top_down4 = lat4
        top_down3 = self.top_down1(lat3) + torch.nn.functional.interpolate(top_down4, scale_factor=2, mode='bilinear',
                                                                           align_corners=False)
        top_down2 = self.top_down2(lat2) + torch.nn.functional.interpolate(top_down3, scale_factor=2, mode='bilinear',
                                                                           align_corners=False)
        top_down1 = self.top_down3(lat1) + torch.nn.functional.interpolate(top_down2, scale_factor=2, mode='bilinear',
                                                                           align_corners=False)

        # 合并特征
        x5 = top_down1

        # 定义一个下采样层，用于调整空间尺寸
        self.downsample = nn.MaxPool2d(kernel_size=16).cuda()

        return self.downsample(x5)


class AF_FPN(nn.Module):
    def __init__(self):
        super(AF_FPN, self).__init__()

        # x1: [2,64,512,512]
        # x2: [2,128,256,256]
        # x3: [2,256,128,128]
        # x4: [2,512,64,64]
        in_channels = [64, 128, 256, 512]
        # 横向连接的1x1卷积，用于调整通道数
        self.lat1 = nn.Conv2d(in_channels[0], 512, kernel_size=1).cuda()
        self.lat2 = nn.Conv2d(in_channels[1], 512, kernel_size=1).cuda()
        self.lat3 = nn.Conv2d(in_channels[2], 512, kernel_size=1).cuda()
        self.lat4 = nn.Conv2d(in_channels[3], 512, kernel_size=1).cuda()

        # 自顶向下连接的3x3卷积，用于上采样
        self.top_down1 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()
        self.top_down2 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()
        self.top_down3 = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()

        # 注意力机制
        self.attention1 = nn.Conv2d(512, 1, kernel_size=1).cuda()
        self.attention2 = nn.Conv2d(512, 1, kernel_size=1).cuda()
        self.attention3 = nn.Conv2d(512, 1, kernel_size=1).cuda()
        self.attention4 = nn.Conv2d(512, 1, kernel_size=1).cuda()

        self.sigmoid = nn.Sigmoid().cuda()

    def forward(self, x1, x2, x3, x4):
        # 横向连接
        lat1 = self.lat1(x1)
        lat2 = self.lat2(x2)
        lat3 = self.lat3(x3)
        lat4 = self.lat4(x4)

        # 自顶向下连接
        top_down4 = lat4
        top_down3 = self.top_down1(lat3) + torch.nn.functional.interpolate(top_down4, scale_factor=2, mode='bilinear',
                                                                           align_corners=False)
        top_down2 = self.top_down2(lat2) + torch.nn.functional.interpolate(top_down3, scale_factor=2, mode='bilinear',
                                                                           align_corners=False)
        top_down1 = self.top_down3(lat1) + torch.nn.functional.interpolate(top_down2, scale_factor=2, mode='bilinear',
                                                                           align_corners=False)

        # 计算注意力权重
        att1 = self.sigmoid(self.attention1(top_down1))
        att2 = self.sigmoid(self.attention2(top_down2))
        att3 = self.sigmoid(self.attention3(top_down3))
        att4 = self.sigmoid(self.attention4(top_down4))

        # 加权融合特征
        weighted_top_down1 = att1 * top_down1
        weighted_top_down2 = att2 * top_down2
        weighted_top_down3 = att3 * top_down3
        weighted_top_down4 = att4 * top_down4

        # weighted_top_down1: [2,512,512,512]
        # weighted_top_down2: [2,512,256,256]
        # weighted_top_down3: [2,512,128,128]
        # weighted_top_down3: [2,512,64,64]

        # 定义一个下采样层，用于调整空间尺寸
        self.downsample1 = nn.MaxPool2d(kernel_size=16).cuda()
        self.downsample2 = nn.MaxPool2d(kernel_size=8).cuda()
        self.downsample3 = nn.MaxPool2d(kernel_size=4).cuda()
        self.downsample4 = nn.MaxPool2d(kernel_size=2).cuda()

        # 最终特征融合
        x5 = self.downsample1(weighted_top_down1) + self.downsample2(weighted_top_down2) + self.downsample3(
            weighted_top_down3) + self.downsample4(weighted_top_down4)

        return x5


def fusion_tensors(x1, x2, x3, x4):
    MFFM_model = MultiScaleFusion()

    # 计算参数数量
    total = sum([param.nelement() for param in MFFM_model.parameters()])
    print("Number of parameter: %.4fM" % (total / 1e6))

    return MFFM_model(x1, x2, x3, x4)


def FPN_fusion(x1, x2, x3, x4):
    fpn_model = FPN()

    # 计算参数数量
    total = sum([param.nelement() for param in fpn_model.parameters()])
    print("Number of parameter: %.4fM" % (total / 1e6))

    return fpn_model(x1, x2, x3, x4)


def AF_FPN_fusion(x1, x2, x3, x4):
    # 创建 AF_FPN 模型的实例
    af_fpn_model = AF_FPN()

    # 计算参数数量
    total = sum([param.nelement() for param in af_fpn_model.parameters()])
    print("Number of parameter: %.4fM" % (total / 1e6))
    return af_fpn_model(x1, x2, x3, x4)


if __name__ == '__main__':
    # 创建一个多尺度融合模块的实例
    # 创建四个随机的tensor作为输入
    x1 = torch.randn(2, 64, 512, 512).cuda()
    x2 = torch.randn(2, 128, 256, 256).cuda()
    x3 = torch.randn(2, 256, 128, 128).cuda()
    x4 = torch.randn(2, 512, 64, 64).cuda()

    output = fusion_tensors(x1, x2, x3, x4)
    # 打印输出的形状，应该是[2.512.32.32]
    print(output.shape)
