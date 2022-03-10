# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# Copy from https://github.com/Halmstad-University/SalsaNext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34, resnet50


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output

class SalResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(SalResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters // 4 + 2 * out_filters, out_filters, (3, 3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class SalsaNext(nn.Module):
    def __init__(self, in_channels=8, nclasses=20, base_channels=32, softmax=True):
        super(SalsaNext, self).__init__()
        self.base_channels = base_channels
        self.dropout_ratio = 0.2
        self.downCntx = ResContextBlock(in_channels, base_channels)
        self.downCntx2 = ResContextBlock(base_channels, base_channels)
        self.downCntx3 = ResContextBlock(base_channels, base_channels)

        self.resBlock1 = SalResBlock(base_channels, 2 * base_channels, self.dropout_ratio, pooling=True, drop_out=False)
        self.resBlock2 = SalResBlock(2 * base_channels, 4 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock3 = SalResBlock(4 * base_channels, 8 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock4 = SalResBlock(8 * base_channels, 8 * base_channels, self.dropout_ratio, pooling=True)
        self.resBlock5 = SalResBlock(8 * base_channels, 8 * base_channels, self.dropout_ratio, pooling=False)

        self.upBlock1 = UpBlock(8 * base_channels, 4 * base_channels, self.dropout_ratio)
        self.upBlock2 = UpBlock(4 * base_channels, 4 * base_channels, self.dropout_ratio)
        self.upBlock3 = UpBlock(4 * base_channels, 2 * base_channels, self.dropout_ratio)
        self.upBlock4 = UpBlock(2 * base_channels, base_channels, self.dropout_ratio, drop_out=False)

        self.logits = nn.Conv2d(base_channels, nclasses, kernel_size=(1, 1))
        self.softmax = softmax

class ResidualBasedFusionBlock(nn.Module):
    def __init__(self, pcd_channels, img_channels):
        super(ResidualBasedFusionBlock, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(pcd_channels + img_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(pcd_channels)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(pcd_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(pcd_channels, pcd_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(pcd_channels),
            nn.Sigmoid()
        )

    def forward(self, pcd_feature, img_feature):
        cat_feature = torch.cat((pcd_feature, img_feature), dim=1)
        fuse_out = self.fuse_conv(cat_feature)
        attention_map = self.attention(fuse_out)
        out = fuse_out * attention_map + pcd_feature
        return out


# ----------------------------------------------------------------------------- #
# Encoder： ResNet
# ----------------------------------------------------------------------------- #

class ResNet(nn.Module):
    def __init__(self, in_channels=3, dropout_rate=0.2,
                 pretrained=True):
        super(ResNet, self).__init__()


        # net = resnet34(pretrained)
        # self.expansion = 1

        net = resnet50(pretrained)
        self.expansion = 4

        self.feature_channels = [64 * self.expansion, 128 * self.expansion, 256 * self.expansion, 512 * self.expansion]

        # Note that we do not downsample for conv1
        # self.conv1 = net.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
        if in_channels == 3:
            self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, img_feature=[]):
        # pad input to be divisible by 16 = 2 ** 4
        h, w = x.shape[2], x.shape[3]
        # check input size
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # inter_features = []
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)  # downsample
        layer3_out = self.dropout(self.layer3(layer2_out))  # downsample
        layer4_out = self.dropout(self.layer4(layer3_out))  # downsample

        return [layer1_out, layer2_out, layer3_out, layer4_out]

class ASPP(nn.Module):  # 更换为 CA 或者 CBAM
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(
            in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(
            image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([
            image_features, atrous_block1, atrous_block6,
            atrous_block12, atrous_block18], dim=1))
        return net


class LiDARAtt(ResNet): ## 改为轻型fusion 参考方法包含： siamese, 4D-net
    def __init__(self, in_channels=5):
        super(LiDARAtt, self).__init__(in_channels=in_channels)

    def forward(self, x):
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        layer1_out = self.layer1(self.maxpool(conv1_out))
        layer2_out = self.layer2(layer1_out)  # downsample
        layer3_out = self.dropout(self.layer3(layer2_out))  # downsample
        layer4_out = self.dropout(self.layer4(layer3_out))  # downsample

        return [layer1_out, layer2_out, layer3_out, layer4_out]

class SalsaNextFusion(SalsaNext):
    def __init__(self, in_channels= 8, nclasses=20, base_channels=32, img_feature_channels=[]):
        super(SalsaNextFusion, self).__init__(in_channels=in_channels, base_channels=base_channels,
                                              nclasses=nclasses, softmax=True)

        self.fusionblock_i1 = ResidualBasedFusionBlock(self.base_channels * 2, img_feature_channels[0])
        self.fusionblock_i2 = ResidualBasedFusionBlock(self.base_channels * 4, img_feature_channels[1])
        self.fusionblock_i3 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[2])
        self.fusionblock_i4 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[3])

        # self.fusionblock_l1 = ResidualBasedFusionBlock(self.base_channels * 2, img_feature_channels[0])
        # self.fusionblock_l2 = ResidualBasedFusionBlock(self.base_channels * 4, img_feature_channels[1])
        # self.fusionblock_l3 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[2])
        # self.fusionblock_l4 = ResidualBasedFusionBlock(self.base_channels * 8, img_feature_channels[3])

        self.aspp = ASPP(self.base_channels * 8, self.base_channels * 8)

    def forward(self, x, img_feature=[], pcd_att=[]):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down0c1 = self.fusionblock_i1(down0c, img_feature[0])
        # down0c2 = self.fusionblock_l1(down0c1, pcd_att[0])
        down0c2 = down0c1

        down1c, down1b = self.resBlock2(down0c2)
        down1c1 = self.fusionblock_i2(down1c, img_feature[1])
        # down1c2 = self.fusionblock_i2(down1c1, pcd_att[1])
        down1c2 = down1c1

        down2c, down2b = self.resBlock3(down1c2)
        down2c1 = self.fusionblock_i3(down2c, img_feature[2])
        # down2c2 = self.fusionblock_i3(down2c1, pcd_att[2])
        down2c2 = down2c1

        down3c, down3b = self.resBlock4(down2c2)
        down3c1 = self.fusionblock_i4(down3c, img_feature[3])
        # down3c2 = self.fusionblock_i4(down3c1, pcd_att[3])
        down3c2 = down3c1

        down5c = self.aspp(self.resBlock5(down3c2))
        # down5c = self.aspp(down3c2)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)
        if self.softmax:
            logits = F.softmax(logits, dim=1)

        return logits


class RGBDecoder(nn.Module):
    def __init__(self, in_channels=[], nclasses=4, base_channels=64):
        super(RGBDecoder, self).__init__()

        self.up_4a = nn.Sequential(
            nn.Conv2d(in_channels[3], base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.up_3a = nn.Sequential(
            nn.Conv2d(in_channels[2] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")

        )
        self.up_2a = nn.Sequential(
            nn.Conv2d(in_channels[1] + base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.up_1a = nn.Sequential(
            nn.Conv2d(in_channels[0] + base_channels, base_channels, 1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(base_channels),
            nn.Upsample(scale_factor=2, mode="bilinear")
        )
        self.conv = nn.Conv2d(base_channels, nclasses, kernel_size=3, padding=1)

    def forward(self, inputs):
        up_4a = self.up_4a(inputs[3])
        up_3a = self.up_3a(torch.cat((up_4a, inputs[2]), dim=1))
        up_2a = self.up_2a(torch.cat((up_3a, inputs[1]), dim=1))
        up_1a = self.up_1a(torch.cat((up_2a, inputs[0]), dim=1))
        out = self.conv(up_1a)
        out = F.softmax(out, dim=1)
        return out


class PMFNet(nn.Module):
    def __init__(self, pcd_channels=5, img_channels=3, nclasses=20, base_channels=32,
                 imagenet_pretrained=True, image_backbone="resnet50"):
        super(PMFNet, self).__init__()

        # self.camera_stream_encoder = ResNet(
        #     in_channels=img_channels,
        #     pretrained=imagenet_pretrained,
        #     backbone=image_backbone)

        self.camera_stream_encoder = ResNet(in_channels=img_channels)
        # self.lidar_stream_att = LiDARAtt(in_channels=pcd_channels)

        # self.camera_stream_decoder_t1 = RGBDecoder(
        #     self.camera_stream_encoder.feature_channels,
        #     nclasses=nclasses, base_channels=self.camera_stream_encoder.expansion*16)

        self.lidar_stream = SalsaNextFusion(in_channels=pcd_channels, nclasses=nclasses,
                                            base_channels=base_channels,
                                            img_feature_channels=self.camera_stream_encoder.feature_channels)


        # self.camera_stream_decoder_t = RGBDecoder(
        #     self.camera_stream_encoder.feature_channels,
        #     nclasses=nclasses, base_channels=self.camera_stream_encoder.expansion*16)

        # self.lidar_stream_att = LiDARAtt(in_channels=pcd_channels,
        #                                  nclasses=nclasses, base_channels=base_channels)

    def forward(self, data):
        img, pcd= data[:,0:3,:,:],data[:,3:7,:,:]
        img_feature = self.camera_stream_encoder(img)
        # pcd_att = self.lidar_stream_att(pcd[:,1,:,:,:])
        # pcd_feature = self.lidar_stream(pcd[:,0,:,:], img_feature)
        # print(pcd_feature.shape)

        # camera_pred = self.camera_stream_decoder(img_feature)
        return img_feature #, pcd_feature  # ,camera_pred


# if __name__ == '__main__':
#     model = PMFNet().cuda()
#     input_pcd, input_img = torch.randn((1, 2, 5, 256, 1024)), torch.randn((1, 3, 256, 1024))
#     out = model(input_pcd.cuda(), input_img.cuda())
#     print(out.shape)
#     # network is huge, only work 1 batch
