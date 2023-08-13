import torch
import torch.nn as nn
from nets.resnet import resnet50
from nets.vgg import VGG16
import timm
import torch.nn.functional as F
# from nets.Enh_net import DenseBlock
# from nets.CANet import CA_Block
# from nets.CBAM import CBAMBlock
# from nets.multiscale_fuse import fusion_tensors
from cv_attention.SequentialSelfAttention import module_1, module_2, module_3, module_4
from nets.multiscale_fuse import fusion_tensors
# from nets.depth_separable_conv import dp_conv


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        # self.conv1 = dp_conv(in_channels=in_size,out_channels=out_size,)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        outputs = torch.cat([inputs1, self.up(inputs2)], 1).to(device)
        outputs = self.conv1(outputs).to(device)
        outputs = self.relu(outputs).to(device)
        outputs = self.conv2(outputs).to(device)
        outputs = self.relu(outputs).to(device)
        return outputs


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1



class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class Unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)

            in_filters = [192, 384, 768, 1024]


        elif backbone == "edgenext" or backbone == "edgenext_xx_small_nodense_ELU":
            self.vgg = timm.create_model('edgenext_xx_small', features_only=True, pretrained=True).to(device)
            in_filters = [192, 384, 768, 1024]

        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # denseblock = _DenseBlock()
        # out=nn.Module(denseblock)

        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        img_ch=3
        t=2
        self.RRCNN1 = RRCNN_block(ch_in=img_ch, ch_out=64, t=t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=t)

        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=t)

        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=t)

        # self.ca_model1 = CA_Block(channel=64, h=512, w=512)
        # self.ca_model2 = CA_Block(channel=128, h=256, w=256)
        # self.ca_model3 = CA_Block(channel=256, h=128, w=128)
        # self.ca_model4 = CA_Block(channel=512, h=64, w=64)
        # self.ca_model5 = CA_Block(channel=512, h=32, w=32)

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.backbone = backbone



    def forward(self, inputs):
        '''VGG forward输出:
                    feat1: torch.size([2,64,512,512])
                    feat2: torch.size([2,128,256,256])
                    feat3: torch.size([2,256,128,128])
                    feat4: torch.size([2,512,64,64])
                    feat5: torch.size([2,512,32,32])
                    '''
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)


        # elif self.backbone == "edgenext_xx_small":
        #     all_feature_extractor = self.vgg
        #     # all_feature_extractor = timm.create_model('resnet34', features_only=True)
        #
        #     all_features = all_feature_extractor(inputs)
        #
        #     # 在这里就对feature 4 denseblock，直接用没处理过的feature 3，减少信息丢失
        #     upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(device)
        #     feature_4 = upsample(all_features[3])
        #
        #     denseblock = DenseBlock(168, 512)
        #     feature_4 = denseblock(feature_4)
        #
        #     all_features.append(feature_4)
        #
        #     upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True).to(device)
        #
        #     # print("\n\n")
        #     # 原始  feature 3 shape: torch.Size([2, 168, 16, 16])
        #     # 调整后feature 3 shape: torch.Size([2, 512, 64, 64])
        #     # feature 4 不参与
        #     for i in range(len(all_features) - 1):
        #         print('feature {} shape: {}'.format(i, all_features[i].shape))
        #
        #         if i == 0:
        #             conv = nn.Conv2d(24, 64, kernel_size=1).to(device)
        #         elif i == 1:
        #             conv = nn.Conv2d(48, 128, kernel_size=1).to(device)
        #         elif i == 2:
        #             conv = nn.Conv2d(88, 256, kernel_size=1).to(device)
        #         elif i == 3:
        #             conv = nn.Conv2d(168, 512, kernel_size=1).to(device)
        #         all_features[i] = conv(all_features[i])
        #         all_features[i] = upsample(all_features[i])
        #
        #     #  print('feature {} shape: {}'.format(i, all_features[i].shape))
        #
        #     # upsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True).to(device)
        #     # all_features.append(upsample(all_features[3]))
        #     print('feature {} shape: {}'.format(4, all_features[4].shape))
        #     [feat1, feat2, feat3, feat4, feat5] = all_features

        elif self.backbone == "edgenext" or self.backbone == "edgenext_xx_small_nodense_ELU":
            all_feature_extractor = self.vgg
            # all_feature_extractor = timm.create_model('resnet34', features_only=True)

            inputs =inputs.to(device)
            all_features = all_feature_extractor(inputs)

            '''EdgeNext输出的tensor
            feature1: [2, 24, 128, 128]
            feature2: [2, 24, 64, 64]
            feature3: [2, 88, 32, 32]
            feature4: [2, 168, 16, 16]
            '''
            # 在这里就对feature 4 denseblock，直接用没处理过的feature 3，减少信息丢失
            # upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(device)


            # denseblock = DenseBlock(168, 512)
            # feature_4 = denseblock(feature_4)

            upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True).to(device)

            # print("\n\n")
            # 原始  feature 3 shape: torch.Size([2, 168, 16, 16])
            # 调整后feature 3 shape: torch.Size([2, 512, 64, 64])

            # feature 4 不参与维度调整
            for i in range(len(all_features)):
                # print('feature {} shape: {}'.format(i, all_features[i].shape))

                if i == 0:
                    conv = nn.Conv2d(24, 64, kernel_size=1).to(device)
                elif i == 1:
                    conv = nn.Conv2d(48, 128, kernel_size=1).to(device)
                elif i == 2:
                    conv = nn.Conv2d(88, 256, kernel_size=1).to(device)
                elif i == 3:
                    conv = nn.Conv2d(168, 512, kernel_size=1).to(device)
                all_features[i] = conv(all_features[i])
                all_features[i] = upsample(all_features[i])

                # print('feature {} shape: {}'.format(i, all_features[i].shape))

            # upsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True).to(device)
            feature_5 = fusion_tensors(all_features[0], all_features[1], all_features[2], all_features[3])
            all_features.append(feature_5)
            # 将feature1-4融合成feature5
            # fuse_feature = fusion_tensors(*all_features[:4])
            # all_features.append(fuse_feature)

            # print('feature {} shape: {}'.format(4, all_features[4].shape))
            [feat1, feat2, feat3, feat4, feat5] = all_features

        # 加注意力机制
        up4 = self.up_concat4(feat4, feat5)
        up4 = module_4(up4)
        # up4 torch.Size([2,512,64,64])
        up3 = self.up_concat3(feat3, up4)
        up3 = module_3(up3)
        # up3 torch.Size([2,256,128,128])
        up2 = self.up_concat2(feat2, up3)
        up2 = module_2(up2)
        # up2 torch.Size([2,128,256,256])
        up1 = self.up_concat1(feat1, up2)
        up1 = module_1(up1)
        # up1 torch.Size([2,64,512,512])

        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "vgg" or self.backbone == "edgenext_x_small":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg" or self.backbone == "edgenext_x_small":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True
