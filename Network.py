import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class MYNET(nn.Module):
    def __init__(self,
                 scale_ratio,
                 n_select_bands,
                 n_bands):
        """Load the pretrained ResNet and replace top fc layer."""
        super(MYNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.n_select_bands = n_select_bands
        self.conv_1st = nn.Sequential(
            nn.Conv2d(n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_bands),
            nn.ReLU(),
        )
        #普通2d卷积 原通道数
        self.conv_2d = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        #亚像素卷积 32到64
        self.conv_ps1_1 = nn.Conv2d(n_bands, 64, (5, 5), (1, 1), (2, 2))
        self.conv_ps1_2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv_ps1_3 = nn.Conv2d(32, 1 * (2 ** 2), (3, 3), (1, 1), (1, 1))
        self.conv_ps2_3 = nn.Conv2d(32, 1 * (4 ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.pixel_shuffle2 = nn.PixelShuffle(4)
        #亚像素卷积 64到128
        #
        #最大池化 128到64
        self.downsample_max_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_select_bands, n_select_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        #平均池化
        self.downsample_avg_1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(n_select_bands, n_select_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # self.conv_tem = nn.Conv2d(2*n_bands+2,n_bands, kernel_size=3, stride=1, padding=1)
        self.conv_xlm = nn.Conv2d(n_bands+1,n_bands+1, kernel_size=3, stride=1, padding=1)
        self.conv_xm = nn.Conv2d(n_bands+2*n_select_bands+1 ,n_bands, kernel_size=3, stride=1, padding=1)
        self.conv_xlh2 = nn.Conv2d(n_bands+1,n_bands+1, kernel_size=3, stride=1, padding=1)
        self.conv_xlh3 = nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1)
        self.conv_xlh = nn.Conv2d(2 * n_bands + 2, n_bands, kernel_size=3, stride=1, padding=1)
        self.conv_xhh = nn.Conv2d(30, n_bands, kernel_size=3, stride=1, padding=1)
        self.conv_group = nn.Conv2d(n_bands, 125, kernel_size=3, stride=1, padding=1)
        self.conv_group1 = nn.Conv2d(5, 4, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(n_bands, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 1 * (2 ** 2), (3, 3), (1, 1), (1, 1))
        self.conv3_1 = nn.Conv2d(32, 1 * (4 ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.pixel_shuffle2 = nn.PixelShuffle(4)
        self.conv4 = nn.Conv2d(n_bands+n_select_bands, n_bands+n_select_bands, kernel_size=3, stride=1, padding=1)#拼接后恢复 亚像素卷积\
        self.conv4_1 = nn.Conv2d(n_bands+2*n_select_bands, n_bands+2*n_select_bands, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(n_bands+1, n_bands+1, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(n_select_bands+1, n_select_bands+1, kernel_size=3, stride=1, padding=1)
        self.conv6_1 = nn.Conv2d(2*n_select_bands+1, 2*n_select_bands+1, kernel_size=3, stride=1, padding=1)
        self.conv_sum = nn.Conv2d((2*(n_bands+n_select_bands)+2), n_bands, kernel_size=3, stride=1, padding=1)
        self.conv_sum_1 = nn.Conv2d((2*(n_bands+2*n_select_bands)+2), n_bands, kernel_size=3, stride=1, padding=1)
        self.poor1 = nn.MaxPool2d(4)
        self.poor2 = nn.MaxPool2d(2)
        self.conv_poor = nn.Conv2d(n_select_bands, n_select_bands, kernel_size=3, stride=1, padding=1)
        self.conv_sum2 = nn.Conv2d((2*(n_bands+n_select_bands)+n_select_bands), n_bands, kernel_size=3, stride=1, padding=1)
        self.conv_sum2_1 = nn.Conv2d((2*(n_bands+2*n_select_bands)+2*n_select_bands), n_bands, kernel_size=3, stride=1, padding=1)

        self.downsample1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(n_select_bands, n_select_bands, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_select_bands),
            nn.ReLU(),
        )
        self.downsample1_1 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(2*n_select_bands, 2*n_select_bands, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_select_bands),
            nn.ReLU(),
        )

        self.downsample2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(n_select_bands, n_select_bands, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_select_bands),
            nn.ReLU(),
        )
        self.downsample2_1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(2*n_select_bands, 2*n_select_bands, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_select_bands),
            nn.ReLU(),
        )
        self.downsample3 = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_select_bands),
            nn.ReLU(),
        )

        self.conv7 = nn.Conv2d(n_bands+n_select_bands+1, n_bands+n_select_bands, kernel_size=3, stride=1, padding=1)
        self.conv7_1 = nn.Conv2d(n_bands+2*n_select_bands+1, n_bands+2*n_select_bands, kernel_size=3, stride=1, padding=1)
        ##

        self.conv_down = nn.Conv2d(n_bands, n_select_bands,kernel_size=3, stride=1, padding=1)
        self.conv_hr = nn.Conv2d(n_select_bands, 2*n_select_bands,kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        a = torch.ones(1, requires_grad=True)
        b = torch.ones(1, requires_grad=True)
        c = torch.ones(1, requires_grad=True)

        self.h_road = nn.Parameter(a)
        self.m_road = nn.Parameter(b)
        self.l_road = nn.Parameter(c)

        a1 = torch.ones(1, requires_grad=True)
        b1 = torch.ones(1, requires_grad=True)
        c1 = torch.ones(1, requires_grad=True)

        self.l_road1 = nn.Parameter(a1)
        self.l_road2 = nn.Parameter(b1)
        self.l_road3 = nn.Parameter(c1)
        self.conv_p = nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1)
    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio/2, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]
        return x_lr
    ####
    # def select_group(self, x_lr, x_hr):
    #     x_lr = self.conv_group(x_lr) #125通道
    #     x_lr = F.interpolate(x_lr, scale_factor=(self.scale_ratio / 2), mode='bilinear')
    #     xl_1 = x_lr[:, 0:25, :, :]
    #     xl_2 = x_lr[:, 25:50, :, :]
    #     xl_3 = x_lr[:, 50:75, :, :]
    #     xl_4 = x_lr[:, 75:100, :, :]
    #     xl_5 = x_lr[:, 100:125, :, :]
    #     return xl_1 ,xl_2, xl_3, xl_4, xl_5
    ####
    def select_group(self, x_lr, x_hr):
        group = []
        for i in range(self.n_bands-5):
            if i == 0:
                x = x_lr[:,0:5,:,:]
                group.append(x)

                 # if i == self.n_bands-1:
                 #    x = x_lr[:,i-4:i+1,:,:]
                 #    group.append(x)

            else:
                x = x_lr[:,i-1:i+4,:,:]
                group.append((x))
        return group

    def forward(self, x_lr, x_hr):
        xl = x_lr
        lr = x_lr  # n,32,32
        hr = x_hr  # 5,128,128
        group = self.select_group(x_lr, x_hr)
        group1 = group[0]
        group1 = F.sigmoid(group1)#
        group1 = self.conv_group1(group1)
        group1 = self.pixel_shuffle1(group1)
        group1 = F.interpolate(group1, scale_factor=(self.scale_ratio/2), mode='bilinear')
        # group2 = self.relu(group1) #

        for i in range(1,self.n_bands-5):
            # group[i] = group[i]
            group[i] = F.sigmoid(group[i])
            group[i] = self.conv_group1(group[i])
            group[i] = self.pixel_shuffle1(group[i])
            group[i] = F.interpolate(group[i], scale_factor=(self.scale_ratio/2), mode='bilinear')
            # group[i] = self.relu(group[i])
            group1 = torch.cat((group1, group[i]), dim=1)
        ##2
        xl0 = F.interpolate(lr, scale_factor=self.scale_ratio, mode='bilinear')  # n,128,128
        xl0 = torch.cat((xl0, hr), dim=1)  # n+5,128,128
        xl0 = self.conv4(xl0)  # n+5,128,128 0通路完成
        # 高光谱低分辨率第1通路
        xl1 = F.tanh(self.conv1(lr))  # 64,32,32
        xl1 = F.tanh(self.conv2(xl1))  # 32，32，32
        xl1 = self.conv3(xl1)  # 4，32，32
        xl1 = F.sigmoid(self.pixel_shuffle1(xl1))  # 1，64，64
        xl1_1 = F.interpolate(lr, scale_factor=self.scale_ratio / 2, mode='bilinear')  # n,64,64
        xl1 = torch.cat((xl1, xl1_1), dim=1)  # n+1,64，64
        xl11 = xl1
        xl1 = F.interpolate(xl1, scale_factor=self.scale_ratio / 2, mode='bilinear')  # n+1,128,128
        xl1 = self.conv5(xl1)  # n+1,128,128 1通路完成
        # 高光谱低分辨率第2通路
        xl2 = F.tanh(self.conv1(lr))
        xl2 = F.tanh(self.conv2(xl2))
        xl2 = self.conv3_1(xl2)  # 16，32，32
        xl2 = F.sigmoid(self.pixel_shuffle2(xl2))  # 1,128,128
        xl2s = xl2
        xl2 = torch.cat((xl2, hr), dim=1)  # 1+5,128,128
        xl2 = self.conv6(xl2)  # 1+5,128,128

        xh0 = self.downsample1(hr)
        xh0 = torch.cat((xh0, lr), dim=1)  # 5+n,32,32
        xh0 = self.conv4(xh0)  # 5+n,32,32
        # 多光谱高分辨率第1通道
        # xh1 = self.poor2(hr) #5,64,64
        # xh1 = self.conv_poor(xh1) #5,64,64
        xh1 = self.downsample2(hr)  # 5，64，64
        xh1 = torch.cat((xh1, xl11), dim=1)  # n+5+1,64,64
        xh1 = self.poor2(xh1)  # n+5+1,32,32
        xh1 = self.conv7(xh1)  # 5+n,32,32
        # 拼接
        xl_out = torch.cat(( self.l_road1*xl0,  self.l_road2*xl1,  self.l_road1*xl2), dim=1)  # n+5+n+1+5+1，128，128
        xl_out = self.conv_sum(xl_out)  # n,128,128
        xh_out = torch.cat((xh0, xh1), dim=1)  # (5+n)*2,32,32
        xh_out = F.interpolate(xh_out, scale_factor=self.scale_ratio, mode='bilinear')  # (5+n)*2,128,128
        xh_out = torch.cat((xh_out, hr), dim=1)  # (5+n)*2+5,128,128
        xh_out = self.conv_sum2(xh_out) + self.conv_2d(self.conv_1st(hr))  # n,128,128
        ##
        x = self.m_road * self.conv_2d(torch.cat((group1, x_hr), dim=1)) + self.l_road * xl_out + self.h_road * xh_out  #all
        #x = self.m_road*torch.cat((group1, x_hr), dim=1) + self.l_road*xl_out + self.h_road*xh_out ##output#######
        # x = self.m_road * self.conv_2d(torch.cat((group1, x_hr), dim=1)) +  self.h_road*xh_out  # group+hr
        # x =   self.m_road*self.conv_2d(torch.cat((group1, x_hr), dim=1)) + self.l_road*xl_out #group+lr
        # x = self.l_road*xl_out + self.h_road*xh_out #L+H
        # x = xl_out #L  up
        # x = xh_out #H down
        # x = self.m_road * self.conv_2d(torch.cat((group1, x_hr), dim=1)) #G ACFE



        return x, 0, 0, 0, 0,  0


