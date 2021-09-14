"""
 > Network architecture of FUnIE-GAN model
   * Paper: arxiv.org/pdf/1903.09766.pdf
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    基础的残差模块
"""
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     #  in_planes 输入通道 planes 输出通道
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.shortcut = nn.Sequential()
#         # 经过处理后的x要与x的维度相同(尺寸和深度)
#         # 如果不相同，需要添加卷积+BN来变换为同一维度
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
#
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out






class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        # layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        # if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        # layers.append(nn.LeakyReLU(0.2))

        if bn:
            # print(bn)
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                # Conv2d的参数out_channels要跟.BatchNorm2d的输入参数要一致。
                nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.8),
                # nn.LeakyReLU(0.2)
            )

            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_size,out_size ,
                #           kernel_size=3, stride=2, bias=False),
                nn.Conv2d(in_size, out_size, 3, 2, 1, 1, bias=False ),
                nn.BatchNorm2d(out_size, momentum=0.8)
            )
            self.l1 = nn.Sequential(
                nn.LeakyReLU(0.2)
            )

        else:
            # 卷积核 4*4 会导致尺寸变成158 238
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                # nn.BatchNorm2d(out_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(out_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(out_size, momentum=0.8),
                # nn.LeakyReLU(0.2)
            )

            self.shortcut = nn.Sequential(
                # nn.Conv2d(in_size,out_size ,
                #           kernel_size=1, stride=2, bias=False),
                nn.Conv2d(in_size, out_size, 3, 2, 1, 1, bias=False),
                # nn.BatchNorm2d(out_size, momentum=0.8)
            )
            self.l1 = nn.Sequential(
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        # setp 1
        out = self.model(x)
        # setp 2
        out2 =self.shortcut(x)
        # feature map add operation
        out.add(out2)
        # relu operation
        out = self.l1(out)

        return out



class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            # nn.Conv2d(in_size,out_size ,
            #           kernel_size=3, stride=2, bias=False),
            nn.Conv2d(in_size, out_size, 3, 1, 1, 1, bias=False),

        )

    def forward(self, x, skip_input):
        x = self.model(x)
        # print("---x" + str(x.size()))
        x = torch.cat((x, skip_input), 1)
        # print("---x2" + str(x.size()))
        x = self.shortcut(x)
        # print("---x3" + str(x.size()))
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        # print()
        self.down2 = UNetDown(32, 64)
        # self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256)
        # self.down5 = UNetDown(256, 512)
        # self.down6 = UNetDown(512, 1024)
        self.down5 = UNetDown(256, 512, bn=False)
        # decoding layers
        self.up1 = UNetUp(512, 256)
        # self.up2 = UNetUp(512, 256)
        # self.up3 = UNetUp(512, 256)
        self.up2 = UNetUp(256, 128)
        self.up3 = UNetUp(128, 64)
        self.up4 = UNetUp(64, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)# 32

        d2 = self.down2(d1)# 64
        d3 = self.down3(d2)# 128
        d4 = self.down4(d3)# 256
        d5 = self.down5(d4)# 512

        u1 = self.up1(d5, d4) # 256

        u2 = self.up2(u1, d3) #256 128
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


class DiscriminatorFunieGAN(nn.Module):
    """ A 4-layer Markovian discriminator as described in the paper
    """
    def __init__(self, in_channels=3):
        super(DiscriminatorFunieGAN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

