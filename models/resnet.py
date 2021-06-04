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


# class res18(nn.Module):
#     def __init__(self, n_classes, method="dann", domain_num=2):
#         super(res18, self).__init__()
#         self.n_classes = n_classes
#         self.method = method
#         self.model = torchvision.models.resnet18(pretrained=True)
#         self.d = self.model.fc.in_features
#         self.model.fc = Identity()

#         self.class_classifier = nn.Sequential()
#         self.class_classifier.add_module('c_fc1', nn.Linear(self.d, 100))
#         self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
#         self.class_classifier.add_module('c_relu1', nn.ReLU(True))
#         self.class_classifier.add_module('c_drop1', nn.Dropout())
#         self.class_classifier.add_module('c_fc3', nn.Linear(100, self.n_classes))

#         if self.method == "dann" or self.method == "cdann":
#             self.domain_classifier = nn.Sequential()
#             self.domain_classifier.add_module('d_fc1', nn.Linear(self.d, 100))
#             self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
#             self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
#             self.domain_classifier.add_module('d_fc2', nn.Linear(100, domain_num))

#             self.class_embeddings = nn.Embedding(self.n_classes, self.d)
    
#     def forward(self, input_data, alpha=0, y=None):
#         # input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
#         feature = self.model(input_data)
#         feature = feature.view(-1, self.d)
#         class_output = self.class_classifier(feature)
#         if self.method == "dann":
#             reverse_feature = ReverseLayerF.apply(feature, alpha)
#             domain_output = self.domain_classifier(reverse_feature)
#             return feature, class_output, domain_output
#         elif self.method == "cdann":
#             if y is None:
#                 return feature, class_output
#             feature = feature + self.class_embeddings(y)
#             reverse_feature = ReverseLayerF.apply(feature, alpha)
#             domain_output = self.domain_classifier(reverse_feature)
#             return feature, class_output, domain_output

#         return feature, class_output

# class res50(nn.Module):
#     def __init__(self, n_classes, method="dann", domain_num=2):
#         super(res50, self).__init__()
#         self.n_classes = n_classes
#         self.method = method
#         self.model = torchvision.models.resnet50(pretrained=True)
#         self.d = self.model.fc.in_features
#         self.model.fc = Identity()

#         self.class_classifier = nn.Sequential()
#         self.class_classifier.add_module('c_fc1', nn.Linear(self.d, 100))
#         self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
#         self.class_classifier.add_module('c_relu1', nn.ReLU(True))
#         self.class_classifier.add_module('c_drop1', nn.Dropout())
#         self.class_classifier.add_module('c_fc3', nn.Linear(100, self.n_classes))

#         if self.method == "dann" or self.method == "cdann":
#             self.domain_classifier = nn.Sequential()
#             self.domain_classifier.add_module('d_fc1', nn.Linear(self.d, 100))
#             self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
#             self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
#             self.domain_classifier.add_module('d_fc2', nn.Linear(100, domain_num))

#             self.class_embeddings = nn.Embedding(self.n_classes, self.d)
    
#     def forward(self, input_data, alpha=0, y=None):
#         feature = self.model(input_data)
#         feature = feature.view(-1, self.d)
#         class_output = self.class_classifier(feature)
#         if self.method == "dann":
#             reverse_feature = ReverseLayerF.apply(feature, alpha)
#             domain_output = self.domain_classifier(reverse_feature)
#             return feature, class_output, domain_output
#         elif self.method == "cdann":
#             if y is None:
#                 return feature, class_output
#             feature = feature + self.class_embeddings(y)
#             reverse_feature = ReverseLayerF.apply(feature, alpha)
#             domain_output = self.domain_classifier(reverse_feature)
#             return feature, class_output, domain_output

#         return feature, class_output

if __name__ == "__main__":
    print(Resnet(n_classes=2, model='resnet18', method='cdann', domain_num=2))