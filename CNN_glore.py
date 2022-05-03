import torch
import torch.nn as nn
from blocks import ConvBlock

'''
vgg like 3D CNN
(CNN-3)
'''

class RNet(nn.Module):
    def __init__(self, in_features, num_class, init='kaimingNormal', dropout=None):
        super(RNet, self).__init__()
        self.conv1 = ConvBlock(in_features=in_features, out_features=16, num=1, pool=True) # /2
        # self.glore = GloRe_Unit(num_features=16, num_nodes=50)
        self.conv2 = ConvBlock(in_features=16, out_features=32, num=2, pool=True) # /4
        #self.glore = GloRe_Unit(num_features=32, num_nodes=50)
        self.conv3 = ConvBlock(in_features=32, out_features=64, num=2, pool=True) # /8
        self.glore = GloRe_Unit(num_features=64, num_nodes=400)
        self.conv4 = ConvBlock(in_features=64, out_features=128, num=2, pool=True) # /16
        # self.glore = GloRe_Unit(num_features=128, num_nodes=50)
        self.conv5 = ConvBlock(in_features=128, out_features=128, num=2, pool=True) # /32
        # self.glore = GloRe_Unit(num_features=128, num_nodes=50)
        self.conv6 = ConvBlock(in_features=128, out_features=128, num=2, pool=True) # /64
        self.dense = nn.Sequential(
            nn.Linear(in_features=1024, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=num_class, bias=True)
        )
        # initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv1 = self.conv1(x)
        # glore = self.glore(conv1)
        conv2 = self.conv2(conv1)
        # glore = self.glore(conv2)
        conv3 = self.conv3(conv2)
        glore = self.glore(conv3)
        conv4 = self.conv4(glore)
        # glore = self.glore(conv4)
        conv5 = self.conv5(conv4)
        # glore = self.glore(conv5)
        conv6 = self.conv6(conv5)
        N, __, __, __, __ = conv6.size()
        out = self.dense(conv6.view(N,-1))
        return out

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h



class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_features, num_nodes, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.normalize = normalize

        # projection map
        self.conv_proj = ConvNd(num_features, self.num_nodes, kernel_size=1)
        
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_features, num_node=self.num_nodes)

        self.blocker = BatchNormNd(num_features, eps=1e-04) 


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)
        x_state_reshaped = x.view(n, self.num_features, -1)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_nodes, -1)
        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_features, *x.size()[2:])

        out = x + self.blocker(x_state)

        return out
