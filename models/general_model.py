import torch.nn as nn
from utils.dann_utils import ReverseLayerF, PositiveLayerF

class CNNModel(nn.Module):

    def __init__(self, num_classes, bn_init, dann=False):
        super(CNNModel, self).__init__()
        self.dann = dann
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        if bn_init:
            self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        if bn_init:
            self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        if bn_init:
            self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        #self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        #self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        #self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, num_classes))

        if self.dann:
            self.domain_classifier = nn.Sequential()
            self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
            self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
            self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
            self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

    def forward(self, input_data, alpha=0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        class_output = self.class_classifier(feature)
        if self.dann:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return feature, class_output, domain_output
        
        return feature, class_output
