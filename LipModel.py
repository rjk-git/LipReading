import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )
        self.shorcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shorcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out)
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        out += self.shorcut(x)
        out = self.relu(out)
        return out


class LipModel(nn.Module):
    def __init__(self, c_in, num_class):
        super(LipModel, self).__init__()
        self.show_log = False

        # 3D卷积
        self.conv3d = nn.Conv3d(c_in, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

        # ResNet
        self.res_channel = 64
        self.resnet = nn.Sequential(
            # 3 4 6 3
            self.ResLayer(64, n_block=1, stride=1),
            self.ResLayer(128, n_block=2, stride=2),
            self.ResLayer(256, n_block=2, stride=2),
            self.ResLayer(512, n_block=2, stride=2)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
        self.res_linear = nn.Linear(self.res_channel, 256)
        self.linear_bn = nn.BatchNorm1d(256)

        # LSTM
        # self.lstm = nn.LSTM(256, 256, num_layers=2, dropout=0.3, bidirectional=True)
        self.lstm = nn.GRU(256, 256, num_layers=1, bidirectional=True)

        # Classfication
        self.classfication = nn.Linear(2*256, num_class)


    def ResLayer(self, out_channel, n_block, stride):
        strides = [stride] + [1] * (n_block-1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.res_channel, out_channel, stride))
            self.res_channel = out_channel
        return nn.Sequential(*layers)


    def forward(self, inputs, targets=None):
        '''

        :param inputs: shape:(bsz, channel, timestep, height, width)
        :param targets: tensor向量
        :return:
        '''

        out = self.conv3d(inputs)
        if self.show_log: print('conv3d:',out.size())
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool3d(out)
        if self.show_log: print('pool3d:', out.size())

        out = torch.transpose(out, 1, 2).contiguous()
        bsz, d, c, h, w = out.size()
        out = out.view(-1, c, h, w)
        if self.show_log: print('resize:', out.size())
        out = self.resnet(out)
        if self.show_log: print('resnet:', out.size())
        out = self.avg_pool(out)
        if self.show_log: print('avg_pool:', out.size())
        out = out.view(bsz*d, self.res_channel)
        if self.show_log: print('resize:', out.size())
        out = self.res_linear(out)
        if self.show_log: print('res_linear:', out.size())
        out = self.linear_bn(out)
        out = out.view(bsz, d, -1)
        if self.show_log: print('linear_bn:', out.size())

        out = out.transpose(0, 1).contiguous()
        out = self.lstm(out)[0]
        out = out.transpose(0, 1)
        if self.show_log: print('lstm:', out.size())

        # _, hidden = self.lstm(out)
        # # print('hidden size:', hidden.size())
        # hidden = hidden.permute(1,0,2).reshape(bsz, -1).contiguous()

        out = self.classfication(out)

        logit = F.softmax(out, -1)
        logit = torch.sum(logit, dim=1)
        result = (logit,)

        # loss
        if torch.is_tensor(targets):
            log_sm = torch.mean(-F.log_softmax(out, -1), dim=1)
            # log_sm = -F.log_softmax(out, -1)
            loss = log_sm.gather(dim=-1, index=targets[:, None]).squeeze()
            result = (logit, loss)

        return result




if __name__ == '__main__':
    lip_net = LipModel(c_in=1, num_class=320)
    a = torch.randn(5, 1, 4, 112, 112)
    out = lip_net(a, None)
    print(out[0].size())