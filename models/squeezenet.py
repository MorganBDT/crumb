''' 
Implementation of SqueezeNet
'''

import torchvision.models as models
import torch.nn as nn

class SqueezeNet(nn.Module):
    
    def __init__(self, model_config):
        self.config = model_config
        super(SqueezeNet, self).__init__()
        
        self.model_layers = models.squeezenet1_0(pretrained=self.config['pretrained'])
        
        # freezing weights for feature extraction if desired
        if self.config['freeze_feature_extract']:
            for param in self.model_layers.parameters():
                param.requires_grad = False
                
        if self.config['n_class'] is not None:
            print("Changing output layer to contain {} classes".format(self.config['n_class']))
            self.model_layers.classifier[1] = nn.Conv2d(512, self.config['n_class'], (3, 3), stride=(1, 1), padding=(1, 1))
            
        self.model = nn.Sequential(self.model_layers, nn.Flatten())
                        
    def forward(self, x):
        out = self.model(x)
        return out
