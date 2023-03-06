import torchvision.models as models
import torch.nn as nn


class SqueezeNetClassifyAfterLayer12(nn.Module):
    def __init__(self, num_classes=None, freeze_feat_extract=False, pretrained=True):
        super(SqueezeNetClassifyAfterLayer12, self).__init__()

        self.model = models.squeezenet1_0(pretrained=pretrained)
        if num_classes is not None:
            print("Changing output layer to contain {} classes".format(num_classes))
            self.model.classifier[1] = nn.Conv2d(512, num_classes, (3, 3), stride=(1, 1), padding=(1, 1))

        if freeze_feat_extract:
            feature_extractor = nn.Sequential(*(list(self.model.features)[:12]))
            for param in feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class SqueezeNetStartAfterLayer12(nn.Module):
    def __init__(self, num_classes=None, core_model=None, pretrained=True):
        super(SqueezeNetStartAfterLayer12, self).__init__()

        if core_model is None:
            self.model = models.squeezenet1_0(pretrained=pretrained)
            print('Changing output layer to contain %d classes.' % num_classes)
            self.model.classifier[1] = nn.Conv2d(512, num_classes, (3, 3), stride=(1, 1), padding=(1, 1))
        else:
            self.model = core_model
        self.cut_layer = 12
        self.block = nn.Sequential(*(list(self.model.features)[self.cut_layer:]),
                                         self.model.classifier, nn.Flatten())

    def forward(self, x):
        return self.block(x)

