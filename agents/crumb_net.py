"""
Implementation of CNN for use with CRUMB
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.parameter import Parameter
from .crumbr import CrumbReconstructor
from utils.random_crop import RandomResizeCrop


class Net(nn.Module):
    def __init__(self, model_config):
        super(Net, self).__init__()
        
        self.config = model_config
        self.batch_size = self.config["batch_size"]

        self.pretrained_weights = self.config['pretrained_weights']
        self.pretrained_dataset_no_of_classes = self.config['pretrained_dataset_no_of_classes']

        self.first_forward_pass = True
        self.initial_memory = None

        # "ResNet18" defaults to squeezenet for backwards compatibility with older code
        if self.config["model_name"] == "SqueezeNet" or self.config["model_name"] == "ResNet18":
            # Load pretrained squeezenet model
            self.model = models.squeezenet1_0(pretrained = self.config['pretrained'])

            # Create function for the early layers
            self.FeatureExtractor = torch.nn.Sequential(*(list(self.model.features)[:self.config["crumb_cut_layer"]]))

            # Determine the dimensions of the feature extractor output, by passing one random "image" into it
            garbage_out = self.FeatureExtractor(torch.randn(1, 3, 224, 224))
            self.compressedChannel = garbage_out.size(1)  # E.g. for crumb_cut_layer=12, this should be 512
            self.origsz = garbage_out.size(2)  # E.g. for crumb_cut_layer=12, this should be 13
            print("Configuring crumb layer for num_feat={}, spatially {}x{}".format(self.compressedChannel, self.origsz, self.origsz))

            if self.pretrained_weights and self.config["model_weights"] is not None:
                no_of_classes = self.pretrained_dataset_no_of_classes
                self.model.classifier[1] = nn.Conv2d(512, no_of_classes, (3, 3), stride=(1, 1), padding=(1, 1))
            else:
                self.model.classifier[1] = nn.Conv2d(512, self.config['n_class'], (3, 3), stride=(1, 1), padding=(1, 1))
            # freezing weights for feature extraction if desired
            for param in self.model.parameters():
                param.requires_grad = True
            #print(self.model)

            # freezing weights for feature extraction if desired
            if self.config['freeze_feature_extract']:
                for param in self.FeatureExtractor.parameters():
                    param.requires_grad = False

            self.block = torch.nn.Sequential(*(list(self.model.features)[self.config["crumb_cut_layer"]:]),
                                             self.model.classifier, torch.nn.Flatten())

        elif self.config["model_name"] == "MobileNet":
            self.compressedChannel = 64
            self.origsz = 14
            self.cutlayer = 8

            self.model = models.mobilenet_v2(pretrained=self.config['pretrained'])
            if self.pretrained_weights:
                self.model.classifier[1] = nn.Linear(1280, self.pretrained_dataset_no_of_classes)
            else:
                self.model.classifier[1] = nn.Linear(1280, self.config['n_class'])

            # freezing weights for feature extraction if desired
            for param in self.model.parameters():
                param.requires_grad = True
            #print(self.model)
            self.FeatureExtractor = torch.nn.Sequential(*(list(self.model.features)[:self.cutlayer]))
            # freezing weights for feature extraction if desired
            if self.config['freeze_feature_extract']:
                for param in self.FeatureExtractor.parameters():
                    param.requires_grad = False

            self.avgpool = torch.nn.Sequential(nn.AvgPool2d(kernel_size=7, stride=1), nn.Flatten()).cuda()
            self.block = torch.nn.Sequential(*(list(self.model.features)[self.cutlayer:]),
                                             self.avgpool,
                                             self.model.classifier)

        else:
            raise ValueError("Invalid model name. Use 'SqueezeNet' or 'MobileNet'")

        if self.config["use_random_resize_crops"]:
            # Create random resize crop. Originally RandomResizeCrop(7, scale=(2 / 7, 1.0)) in REMIND paper, using
            # 7x7 (spatially) feature maps. This roughly preserves the 2/7 ratio for other square feature map sizes.
            self.random_resize_crop = RandomResizeCrop(self.origsz, scale=(int(round(self.origsz*(2.0/7.0))) / self.origsz, 1.0))

        if self.config["storage_type"] in ["image", "raw_feature", "merec"] or (self.config["pretrained_weights"] and self.config["memory_weights"] is not None):
            # PretrainedCRUMB matrix will be used, initialize to zeros (to be replaced)
            print("InitializingCRUMB to all zeros")
            mem_init_strat = "zeros"
        else:
            mem_init_strat = self.config["memory_init_strat"]
            print("Initializing newCRUMB matrix using this strategy: " + mem_init_strat)

        if mem_init_strat in ["random_everyperm", "random_distmatch_dense", "random_distmatch_sparse", "random_everyperm_shuffled", "random_everyperm_remove_half_rows"]:
            if self.config["crumb_cut_layer"] == 12:
                fbanks_std = 14
                fbanks_sparsity = 0.64
            elif self.config["crumb_cut_layer"] == 3:
                fbanks_std = 0.73
                fbanks_sparsity = 0.25
            else:
                raise ValueError("Please define the feature banks standard deviation for crumb_cut_layer = " + str(self.config["crumb_cut_layer"]) + ". Enter a dummy value initially, and then check the output of mem_stats in the log to see the fbank standard deviation.")
        else:
            fbanks_std = None
            fbanks_sparsity = None

        self.crumbr = CrumbReconstructor(self.compressedChannel, self.origsz, self.origsz, mem_init=mem_init_strat,
                                         n_memblocks=self.config['n_memblocks'], memblock_length=self.config['memblock_length'], fbanks_std=fbanks_std, fbanks_sparsity=fbanks_sparsity)

        if self.config['freeze_memory']:
            self.crumbr.memory.requires_grad = False
        else:
            self.crumbr.memory.requires_grad = True

        self.initial_memory = torch.clone(self.crumbr.memory.data).detach().cpu()

    def forward(self, x, get_mem_stats=False, use_random_resize_crops=False, storage_type="feature", test_perturbation=None):
        extracted = self.FeatureExtractor(x)

        if test_perturbation is None:
            pass
        elif test_perturbation == "spatial":
            ex_copy = torch.clone(extracted)
            # Generate a stack of all possible indices in 2D
            inds_list = []
            for x in range(extracted.size(2)):
                inds = torch.zeros((extracted.size(2), 2))
                inds[:, 0] = x
                inds[:, 1] = torch.arange(0, extracted.size(2))
                inds_list.append(inds)
            all_inds = torch.vstack(inds_list)
            for im_ind in range(extracted.size(0)):  # Shuffle each image separately
                shuffle_inds = torch.clone(all_inds)
                idx = torch.randperm(shuffle_inds.size(0))
                shuffle_inds = shuffle_inds[idx, :]
                counter = 0
                for x in range(extracted.size(2)):
                    for y in range(extracted.size(3)):
                        extracted[im_ind, :, x, y] = ex_copy[im_ind, :, shuffle_inds.long()[counter, 0], shuffle_inds.long()[counter, 1]]
                        counter = counter + 1
        elif test_perturbation == "feature":
            proportion_zeroed = 0.5
            k = int(round(proportion_zeroed * extracted.size(1)))
            for im_ind in range(extracted.size(0)):  # Shuffle each image separately
                idx = torch.randperm(extracted.size(1))
                idx = idx[:k]
                extracted[im_ind, idx, :, :] = 0
        elif test_perturbation == "spatial_shuffle_rows_cols": # Shuffle all rows, then all cols (fast but incomplete spatial shuffling)
            for im_ind in range(extracted.size(0)):  # Shuffle each image separately
                idx1 = torch.randperm(extracted.size(2))
                extracted[im_ind, :, :, :] = extracted[im_ind, :, idx1, :]
                idx2 = torch.randperm(extracted.size(3))
                extracted[im_ind, :, :, :] = extracted[im_ind, :, :, idx2]
        elif test_perturbation == "feature_complete":
            for im_ind in range(extracted.size(0)):  # Shuffle each image separately
                idx = torch.randperm(extracted.size(1))
                extracted[im_ind, :, :, :] = extracted[im_ind, idx, :, :]
        elif test_perturbation == "feature_swap":
            proportion_randomized = 0.5
            k = int(round(proportion_randomized * extracted.size(1)))
            for im_ind in range(extracted.size(0)):  # Shuffle each image separately
                idx1 = torch.randperm(extracted.size(1))
                idx1 = idx1[:k]
                idx2 = torch.randperm(extracted.size(1))
                idx2 = idx2[:k]
                extracted[im_ind, idx1, :, :] = extracted[im_ind, idx2, :, :]

        direct_logits = self.block(extracted)

        if storage_type == "feature":

            if self.first_forward_pass and self.config["memory_init_strat"] == "sample_first_batch" and self.config["memory_weights"] is None:
                with torch.no_grad():
                    print("Replacing codebook memory block matrix with values randomly drawn from the first image batch...")
                    x_np = np.reshape(extracted.clone().detach().cpu().numpy(), -1)
                    sample = np.random.choice(x_np, (self.crumbr.n_memblocks, self.crumbr.memblock_length))
                    self.crumbr.memory.data = torch.from_numpy(np.copy(sample)).to('cuda')
                    self.initial_memory = Parameter(torch.from_numpy(np.copy(sample)).to('cuda'))
                if self.config['freeze_memory']:
                    self.crumbr.memory.requires_grad = False
                else:
                    self.crumbr.memory.requires_grad = True
                self.initial_memory.requires_grad = False
            self.first_forward_pass = False

            read = self.crumbr.reconstruct_fbank_top1(extracted)

            # Random resize cropping: feature-level data augmentation.
            # Adapted from https://github.com/tyler-hayes/REMIND/blob/master/image_classification_experiments/REMINDModel.py
            if use_random_resize_crops:
                extracted_batch_transformed = torch.empty_like(extracted)
                for tens_ix, tens in enumerate(read):
                    extracted_batch_transformed[tens_ix] = self.random_resize_crop(tens)
                extracted = extracted_batch_transformed

                read_batch_transformed = torch.empty_like(read)
                for tens_ix, tens in enumerate(read):
                    read_batch_transformed[tens_ix] = self.random_resize_crop(tens)
                read = read_batch_transformed
                read = self.crumbr.reconstruct_fbank_top1(read)

            crumb_logits = self.block(read)

            if get_mem_stats:
                mem_stats = self.crumbr.mem_stats(extracted, read)
            else:
                mem_stats = None
        elif storage_type in ["image", "raw_feature", "enhanced_raw_feature", "merec"]:
            crumb_logits = direct_logits
            mem_stats = None
        else:
            raise ValueError("Invalid 'storage_type': " + str(storage_type))
        
        return direct_logits, crumb_logits, mem_stats

    def images_to_storage(self, image_batch, storage_type="feature"):

        extracted = self.FeatureExtractor(image_batch)

        if storage_type == "feature":
            storage_1_example_per_row = self.crumbr.top1_inds_to_replay_storage(self.crumbr.fbank_to_top1_inds(extracted))
        elif storage_type == "image":
            storage_1_example_per_row = image_batch.view(image_batch.size(0), -1)
        elif storage_type in ["raw_feature", "merec"]:
            storage_1_example_per_row = extracted.view(extracted.size(0), -1)
        else:
            raise ValueError("Invalid 'storage_type': " + str(storage_type))

        stored_examples = storage_1_example_per_row.detach().cpu()

        return stored_examples
     
    def forward_direct_only(self, x):
        extracted = self.FeatureExtractor(x)
        direct_logits = self.block(extracted)
        
        return direct_logits
    
    def forward_from_fbanks(self, read):
        crumb_logits = self.block(read)
        
        return crumb_logits
        
    def evalModeOn(self):
        self.eval()
        self.FeatureExtractor.eval()
        self.block.eval()
        self.crumbr.memory.requires_grad = False
    
    def trainModeOn(self):
        self.train()
        self.FeatureExtractor.train() 
        self.block.train()
        if self.config['freeze_feature_extract']:
            for param in self.FeatureExtractor.parameters():
                param.requires_grad = False
       
        self.crumbr.memory.requires_grad = False
        
            
    def trainMemoryOn(self):
        self.train()
        self.FeatureExtractor.train() 
        self.block.train()
        for param in self.FeatureExtractor.parameters():
            param.requires_grad = False
            
        for param in self.block.parameters():
            param.requires_grad = True
       
        self.crumbr.memory.requires_grad = True
        
    def trainOverallOn(self):
        self.train()
        self.FeatureExtractor.train() 
        self.block.train()
        if self.config['freeze_feature_extract']:
            for param in self.FeatureExtractor.parameters():
                param.requires_grad = False
            
        for param in self.block.parameters():
            param.requires_grad = True
       
        if self.config['freeze_memory']:
            self.crumbr.memory.requires_grad = False
        else:
            self.crumbr.memory.requires_grad = True
                


