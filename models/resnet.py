# import torch
# import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision

# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
#     'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#     'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
#     'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
#     'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
# }

# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
        
#     def forward(self, x):
#         return x


# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
# def conv1x1(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=1, bias=False)

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.downsample = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )
    
#     def forward(self, x):
#         t = self.conv1(x)
#         out = F.relu(self.bn1(t))
#         t = self.conv2(out)
#         out = self.bn2(self.conv2(out))   
#         t = self.downsample(x)
#         out += t        
#         out = F.relu(out)
        
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=2):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         # self.conv1 = conv3x3(3,64)
#         # for large input size
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
#                                bias=False)
                        
#         self.bn1 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # end 
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(512*block.expansion, num_classes)
        
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out= self.maxpool(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return out, y
        
#         # function to extact the multiple features
#     def feature_list(self, x):
#         out_list = []
#         out = F.relu(self.bn1(self.conv1(x)))
#         out= self.maxpool(out)
#         out_list.append(out)
#         out = self.layer1(out)
#         out_list.append(out)
#         out = self.layer2(out)
#         out_list.append(out)
#         out = self.layer3(out)
#         out_list.append(out)
#         out = self.layer4(out)
#         out_list.append(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         y = self.linear(out)
#         return y, out_list
    
#     # function to extact a specific feature
#     def intermediate_forward(self, x, layer_index):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out= self.maxpool(out)
#         if layer_index == 1:
#             out = self.layer1(out)
#         elif layer_index == 2:
#             out = self.layer1(out)
#             out = self.layer2(out)
#         elif layer_index == 3:
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#         elif layer_index == 4:
#             out = self.layer1(out)
#             out = self.layer2(out)
#             out = self.layer3(out)
#             out = self.layer4(out)               
#         return out

#     def load(self, path="resnet_svhn.pth"):
#         tm = torch.load(path, map_location="cpu")        
#         self.load_state_dict(tm)
    
# # https://neurohive.io/wp-content/uploads/2019/01/resnet-architectures-34-101.png 
# format = {'resnet18':[2,2,2,2], 'resnet34':[3,4,6,3], 'resnet50':[3,4,6,3]}


# def load_model(pretrained = False, arch='resnet18'):
#     '''
#     load resnet
#     '''

#     # torch_model = torchvision.models.resnet18(pretrained=False)
#     # return model
#     if arch=='resnet18':
#         torch_model = ResNet(BasicBlock, format[arch], num_classes=2)
#         if pretrained:
#             model_dict = torch_model.state_dict()
#             pretrained_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
#                                                 progress = True)
#             # 1. filter out unnecessary keys
#             pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#             # 2. overwrite entries in the existing state dict
#             model_dict.update(pretrained_dict) 
#             # 3. load the new state dict
#             torch_model.load_state_dict(model_dict)
#         print("ResNet Loading Done")
#     if arch=='resnet50gdro':
#         torch_model = torchvision.models.resnet50(pretrained=False)
#         path = 
#         checkpoint = torch.load(PATH)


#     return torch_model




# if __name__ == "__main__":
#     load_model(True)

import torchvision
import torch.nn as nn

from utils.dann_utils import ReverseLayerF
        
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Resnet(nn.Module):
    def __init__(self, n_classes, model='resnet18', method="dann", domain_num=2):
        super(Resnet, self).__init__()
        self.n_classes = n_classes
        self.method = method

        if model == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
        elif model == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
        else:
            raise NotImplementedError
        self.d = self.model.fc.in_features
        self.model.fc = Identity()

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.d, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc3', nn.Linear(100, self.n_classes))

        if self.method == "dann" or self.method == "cdann":
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(self.d, 100))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(100, domain_num))

            self.class_embeddings = nn.Embedding(self.n_classes, self.d)
    
    def forward(self, input_data, alpha=0, y=None):
        feature = self.model(input_data)
        feature = feature.view(-1, self.d)
        class_output = self.class_classifier(feature)
        if self.method == "dann":
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return feature, class_output, domain_output
        elif self.method == "cdann":
            if y is None:
                return feature, class_output
            feature = feature + self.class_embeddings(y)
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return feature, class_output, domain_output

        return feature, class_output
    
def load_model(pretrained = False, arch='resnet18'):
    if arch == 'resnet18':
        return Resnet(n_classes=2, model='resnet18', method='none', domain_num=2)
    elif arch == 'resnet50':
        return Resnet(n_classes=2, model='resnet50', method='none', domain_num=2)

if __name__ == "__main__":
    print(Resnet(n_classes=2, model='resnet18', method='cdann', domain_num=2))
