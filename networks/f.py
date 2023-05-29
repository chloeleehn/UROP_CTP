import torch
from torch import nn
from networks.cnn_blocks import *


def create_model_F(input_channel, n_filters = 16):

    net = F_model(
        n_channels=input_channel, 
        n_filters=n_filters,
        normalization='batchnorm')
    
    model = net.cuda()
    model.initialize_weights()
    return model

class F_model(nn.Module):
    def __init__(self, n_channels=64, n_filters=16, normalization='none'):
        super(F_model, self).__init__()
        convBlock = ResidualConvBlock

        self.conv_start = ConvBlock(1, n_channels , n_filters, normalization=normalization)


        self.block_one = convBlock(2, n_filters , n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, n_filters * 2, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(2, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(2, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(2, n_filters * 16, n_filters * 16, normalization=normalization)


        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(2, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(2, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(2, n_filters, n_filters, normalization=normalization)

        self.final_conv2d = nn.Conv2d(n_filters, 1, 1, padding=0)

        self.sig = torch.nn.Sigmoid()

    def encoder(self, input, fuse_feat):
        
        # fuse-feat: Features from encoder E, we fuse layer-by-layer
        enc_g_x1 = fuse_feat[0]
        enc_g_x2 = fuse_feat[1]
        enc_g_x3 = fuse_feat[2]
        enc_g_x4 = fuse_feat[3]
        enc_g_x5 = fuse_feat[4]

        x1 = self.conv_start(input)

        x1 = self.block_one(x1)
        x1_fused = (x1 + enc_g_x1)
        x1_dw = self.block_one_dw(x1_fused)

        x2 = self.block_two(x1_dw)
        x2_fused = (x2 + enc_g_x2)
        x2_dw = self.block_two_dw(x2_fused)

        x3 = self.block_three(x2_dw)
        x3_fused = (x3 + enc_g_x3)
        x3_dw = self.block_three_dw(x3_fused)

        x4 = self.block_four(x3_dw)
        x4_fused = (x4 + enc_g_x4)
        x4_dw = self.block_four_dw(x4_fused)

        x5 = self.block_five(x4_dw)
        x5_fused = (x5 + enc_g_x5)
        
        res = [x1_fused, x2_fused, x3_fused, x4_fused, x5_fused]
        #res = [x1, x2, x3, x4, x5]
        return res

    def decoder(self, features):
        x1_fused = features[0]
        x2_fused = features[1]
        x3_fused = features[2]
        x4_fused = features[3]
        x5_fused = features[4]
        
        x5_up = self.block_five_up(x5_fused)
        x5_up = x5_up + x4_fused

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3_fused

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2_fused

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1_fused
        
        x9 = self.block_nine(x8_up)

        out = self.final_conv2d(x9)

        return out


    def forward(self, input, fuse_feat):
        features = self.encoder(input, fuse_feat)
        out_seg = self.decoder(features)
        return out_seg

    def initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
