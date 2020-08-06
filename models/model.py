"""DANN model."""

import torch.nn as nn
from .functions import ReverseLayerF
from torchvision import models
from .alexnet import alexnet


class Classifier(nn.Module):
    """ SVHN architecture without discriminator"""

    def __init__(self):
        super(Classifier, self).__init__()
        self.restored = False

        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(1, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 1, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        class_output = self.class_classifier(feature)

        return class_output


class MNISTmodel(nn.Module):
    """ MNIST architecture
    +Dropout2d, 84% ~ 73%
    -Dropout2d, 50% ~ 73%
    """

    def __init__(self):
        super(MNISTmodel, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            nn.BatchNorm2d(48),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.restored = False
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.classifier.add_module('c_relu1', nn.ReLU(True))
        self.classifier.add_module('c_drop1', nn.Dropout2d())
        self.classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.classifier.add_module('c_relu2', nn.ReLU(True))
        self.classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.discriminator = nn.Sequential()
        self.discriminator.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.discriminator.add_module('d_bn1', nn.BatchNorm1d(100))
        self.discriminator.add_module('d_relu1', nn.ReLU(True))
        self.discriminator.add_module('d_fc2', nn.Linear(100, 2))
        self.discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output


class LiptonCNNModel(nn.Module):
    def __init__(self):
        super(LiptonCNNModel, self).__init__()
        self.restored = False
        self.feature = nn.Sequential(
                        nn.Conv2d(3, 6, kernel_size=5, padding=2),nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        nn.Flatten(),
                        nn.Linear(16*5*5, 120), nn.Sigmoid(),
                        nn.Linear(120, 84), nn.Sigmoid())
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(84, 10))
        self.classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.discriminator = nn.Sequential()
        self.discriminator.add_module('d_fc1', nn.Linear(84, 500))
        self.discriminator.add_module('d_fc2', nn.Linear(500,500))
        self.discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))
        
    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 84)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)
        return class_output, domain_output

class MNISTmodel_plain(nn.Module):
    """ MNIST architecture
    +Dropout2d, 84% ~ 73%
    -Dropout2d, 50% ~ 73%
    """

    def __init__(self):
        super(MNISTmodel_plain, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=(5, 5)),  # 3 28 28, 32 24 24
            #nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 32 12 12
            nn.Conv2d(in_channels=32, out_channels=48,
                      kernel_size=(5, 5)),  # 48 8 8
            #nn.BatchNorm2d(48),
            #nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 48 4 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(48*4*4, 100),
            #nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            #nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(48*4*4, 100),
            #nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 48 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output


class SVHNmodel(nn.Module):
    """ SVHN architecture
    """

    def __init__(self):
        super(SVHNmodel, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5)),  # 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 13
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5)),  # 9
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),  # 4
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4)),  # 1
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128 * 1 * 1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, input_data, alpha = 1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128 * 1 * 1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output


class GTSRBmodel(nn.Module):
    """ GTSRB architecture
    """

    def __init__(self):
        super(GTSRBmodel, self).__init__()
        self.restored = False

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(5, 5), stride=1, padding=2),  # 36 ; 44
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 18 ; 22
            nn.Conv2d(in_channels=96, out_channels=144, kernel_size=(3, 3), stride=1, padding=1),  # 16 ; 20
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 8 ; 10
            nn.Conv2d(in_channels=144, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),  # 4 ; 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # 2 ; 3
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 43),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(256 * 5 * 5, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 2),
        )

    def forward(self, input_data, alpha = 1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 40, 40)
        feature = self.feature(input_data)
        feature = feature.view(-1, 256 * 5 * 5)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output


class AlexModel(nn.Module):
    """ AlexNet pretrained on imagenet for Office dataset"""

    def __init__(self):
        super(AlexModel, self).__init__()
        self.restored = False
        model_alexnet = models.alexnet(pretrained=True)

        self.features = model_alexnet.features

        self.fc = nn.Sequential()
        for i in range(6):
            self.fc.add_module("classifier" + str(i),
                               model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features  # 4096

        self.bottleneck = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048, 31),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 2),
        )

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 227, 227)
        feature = self.features(input_data)
        feature = feature.view(-1, 256*6*6)
        fc = self.fc(feature)
        bottleneck = self.bottleneck(fc)

        reverse_bottleneck = ReverseLayerF.apply(bottleneck, alpha)

        class_output = self.classifier(bottleneck)
        domain_output = self.discriminator(reverse_bottleneck)

        return class_output, domain_output

class SALADModel(nn.Module):
    """ SALAD architecture (deep convolutional)
    """

    def __init__(self):
        super(SALADModel, self).__init__()
        self.restored = False
        self.feature = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d((2,2)), 
            nn.Dropout(), 
            
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Dropout(),
            
            nn.Conv2d(256, 512, (3, 3), padding=0),
            nn.BatchNorm2d(512),  
            nn.ReLU(),
            nn.Conv2d(512, 256, (1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (1, 1), padding=1),
            nn.BatchNorm2d(128), 
            nn.AvgPool2d(6)
        )


        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

        self.discriminator = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, input_data, alpha = 1.0):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 128)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.classifier(feature)
        domain_output = self.discriminator(reverse_feature)

        return class_output, domain_output

# class DCGANModel(nn.Module):
#     """ SALAD architecture (deep convolutional)
#     """

#     def __init__(self):
#         super(SALADModel, self).__init__()
#         ngf = 64 
#         self.feature = nn.Sequential(
#             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
 
            
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(256 * 5 * 5, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(512, 43),
#         )

#         self.discriminator = nn.Sequential(
#             nn.Linear(256 * 5 * 5, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(1024, 2),
#         )

#     def forward(self, input_data, alpha = 1.0):
#         input_data = input_data.expand(input_data.data.shape[0], 3, 40, 40)
#         feature = self.feature(input_data)
#         feature = feature.view(-1, 256 * 5 * 5)
#         reverse_feature = ReverseLayerF.apply(feature, alpha)
#         class_output = self.classifier(feature)
#         domain_output = self.discriminator(reverse_feature)

#         return class_output, domain_output


