"""
Implementation of CRUMB strategy to combat catastrophic forgetting
"""

import os
import sys
import random
import math
import numpy as np
import torch
import torch.nn as nn
from .crumb_net import Net
from utils.metric import accuracy, AverageMeter, Timer
import scipy.io as sio
from welford import Welford


class Crumb(nn.Module):
    """
    Normal Neural Network with SGD for classification
    """
    def __init__(self, agent_config):
        """
        Parameters
        ----------
        agent_config : {
            lr = float, momentum = float, weight_decay = float,
            model_weights = str,
            gpuid = [int]
            }
        """
        super(Crumb, self).__init__()
        
        self.config = agent_config
        
        # print function
        self.log = print
        
        # define memory
        self.n_memblocks = agent_config['n_memblocks']
        self.memblock_length = agent_config['memblock_length']
        self.capacity = agent_config['memory_size']
        
        # create the model
        self.net = self.create_model()

        # define the loss function
        self.criterion_fn = nn.CrossEntropyLoss()
        
        self.used_mem_slots = torch.zeros(self.n_memblocks)

        if self.config['visualize']:
            # for visualization purpose
            # store some examples of reading attention
            self.viz_read_att = {i: [] for i in range(agent_config['n_class'])}
            self.viz_NumEgs = 50 #number of examples to store for visualization

        # gpu config
        if agent_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
            self.memory_storage = {} #dictionary: storing reading attentions
        else:
            self.gpu = False
            self.memory_storage = {} #dictionary: storing reading attentions
        
        # initialize the optimizer
        self.init_optimizer()
        
        # denotes which output nodes are active for each task
        self.active_out_nodes = list(range(agent_config['n_class']))
        
    # creates desired model   
    def create_model(self):
        
        cfg = self.config

        torch.cuda.set_device(cfg['gpuid'][0])
        
        # Define the backbone (MLP, LeNet, VGG, ResNet, etc)
        # Model type: MLP, ResNet, etc
        # Model name: MLP100, MLP1000, etc
        # We used modified backbones because of memory, reading, and writing head
        net = Net(cfg)
        
        # load pretrained weights if specified
        #eg. cfg['model_weights'] = 'model_pretrained = './pretrained_model/'
        #print('model_weights')
        if cfg['model_weights'] is not None:
            print('=> Load model weights: '+  cfg['model_weights'])
            # loading weights to CPU
            netPath = cfg['model_weights'] + '_net.pth'
            if torch.cuda.is_available():
                preweights = torch.load(netPath, map_location=torch.device("cuda"))
            else:
                print("WARNING: loading weights onto CPU (without CUDA)")
                preweights = torch.load(netPath, map_location=torch.device('cpu'))
            net.load_state_dict(preweights, strict=False)

            if not cfg["visualize"] and not cfg["continuing"]:
                if cfg["model_name"] == "SqueezeNet":
                    if self.config["crumb_cut_layer"] == 12:
                        net.block[1][1] = nn.Conv2d(net.compressedChannel, self.config['n_class'], (3, 3), stride=(1, 1), padding=(1, 1))
                    elif self.config["crumb_cut_layer"] == 3:
                        net.block[-2][1] = nn.Conv2d(512, self.config['n_class'], (3, 3), stride=(1, 1), padding=(1, 1))
                    else:
                        print("Figure out where in the model you need to make your classification layer")
                        raise NotImplementedError
                elif cfg["model_name"] == "MobileNet":
                    net.block[12][1] = nn.Linear(1280, self.config['n_class'])
                else:
                    raise ValueError("Need to account for newly added model type (i.e. not SqueezeNet or MobileNet) here. Try printing net.block to figure out which layer you need to replace")
            print("=> Load done")
            
        if cfg['memory_weights'] is not None:
            print('=> Load memory weights: '+  cfg['memory_weights'])
            # loading weights to CPU
            netPath = cfg['memory_weights'] + '_mem.pth'
            if torch.cuda.is_available():
                preweights = torch.load(netPath, map_location=torch.device("cuda"))
            else:
                print("WARNING: loading weights onto CPU (without CUDA)")
                preweights = torch.load(netPath, map_location=torch.device('cpu'))

            net.crumbr.memory = preweights
            print("=> Load done")

        # if cfg["storage_type"] == "image":
        #     print("Using images for replay - setCRUMB to all zeros as a sanity check")
        #     net.crumbr.memory.fill_(0)
        #     print("Mean of codebook after setting to zero: " + str(net.crumbr.memory.mean()))
        
        return net
    
    # initialize optimzer
    def init_optimizer(self):
        optimizer_arg_net = {'params': self.net.parameters(),
                         'lr': self.config['lr'],
                         'weight_decay': self.config['weight_decay']}        
        
        if self.config['optimizer'] in ['SGD', 'RMSprop']:
            optimizer_arg_net['momentum'] = self.config['momentum']
            
        elif self.config['optimizer']  in ['Rprop']:
            optimizer_arg_net.pop['weight_decay']
            
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg_net['amsgrad'] == True            
            self.config['optimizer'] = 'Adam'
            
        self.optimizer_net = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg_net)
    
    # make a prediction
    def predict(self, inputs, test_perturbation=None):
        direct, out, _ = self.net.forward(inputs, storage_type=self.config["storage_type"], test_perturbation=test_perturbation)
        return direct.detach().cpu(), out.detach().cpu()
    
    # calculate loss
    def criterion_classi(self, pred, target):
        # mask inactive output nodes
        pred = pred[:,self.active_out_nodes]
        loss = self.criterion_fn(pred, target)
        return loss

    def merec_sample(self, w):  # w is a Welford object from the welford python library
        mean = w.mean
        # return torch.Tensor(np.random.multivariate_normal(w.mean, np.diag(w.var_p)))
        return torch.Tensor(mean + np.random.randn(mean.shape[0]) * np.sqrt(w.var_p))

    # replay old reading attention
    def criterion_replay(self, target, storage_type="feature"):
        labelslist = target.tolist()

        replay_labels = [cls for cls in self.active_out_nodes
                         if cls not in labelslist and cls in self.memory_storage.keys()]
        replay_batchsize = len(replay_labels)
        
        if replay_batchsize > 0:
            
            replaytimes = math.ceil(len(labelslist)/replay_batchsize)
            print("REPLAY BATCH SIZE: " + str(replaytimes * replay_batchsize))
            replay_examples_list = []

            # find corresponding memory block indices for replaying classes
            for i in range(replaytimes):
                if storage_type == "merec":
                    temp_store = [self.merec_sample(self.memory_storage[i]) for i in replay_labels[0:replay_batchsize]]
                else:
                    temp_store = [self.memory_storage[i][random.randint(0, self.memory_storage[i].size(0) - 1)] for i in replay_labels[0:replay_batchsize]]
                replay_examples_list += temp_store
                replay_labels += replay_labels[0:replay_batchsize]
                
            replay_labels = replay_labels[0:-replay_batchsize]
            replay_labels = torch.LongTensor(replay_labels)
            
            # Convert to tensor for training
            replay_examples = torch.stack(replay_examples_list)
            if storage_type == "image":
                replay_read = replay_examples.view(-1, 3, 224, 224)
            elif storage_type in ["raw_feature", "enhanced_raw_feature", "merec"]:
                replay_read = replay_examples.view(-1, self.net.compressedChannel, self.net.origsz, self.net.origsz)
            elif storage_type == "feature":
                replay_read = self.net.crumbr.top1_inds_to_fbank(replay_examples.long())
            else:
                raise ValueError("Invalid storage type: '" + str(storage_type) + "'")

            if self.gpu:
                replay_labels = replay_labels.cuda()
                replay_read = replay_read.cuda()
                
            # replay training starts
            self.optimizer_net.zero_grad()

            # Random resize cropping: feature-level data augmentation.
            # Adapted from https://github.com/tyler-hayes/REMIND/blob/master/image_classification_experiments/REMINDModel.py
            if self.config["use_random_resize_crops"]:
                read_batch_transformed = torch.empty_like(replay_read)
                for tens_ix, tens in enumerate(replay_read):
                    read_batch_transformed[tens_ix] = self.net.random_resize_crop(tens)
                replay_read = read_batch_transformed
                replay_read = self.net.crumbr.reconstruct_fbank_top1(replay_read)

            if self.config['augment_replays']:
                for i in range(list(replay_read.shape)[0]):
                    if random.random() < 0.5:
                        replay_read[i, :, :, :] = torch.flip(replay_read[i, :, :, :], [1])
            if storage_type == "image":
                logits = self.net.forward_direct_only(replay_read)
            elif storage_type in ["feature", "raw_feature", "enhanced_raw_feature", "merec"]:
                logits = self.net.forward_from_fbanks(replay_read)
            else:
                raise ValueError("Invalid storage type: '" + str(storage_type) + "'")

            replayloss = self.config['replay_coef']*self.criterion_classi(logits, replay_labels)
            replayloss.backward()
            self.clip_grads(self.net)
            self.optimizer_net.step() 
            
            return replay_batchsize, replayloss
        else:
            return 0, float('NaN')

    def get_saliency(self, feature_map_batch, target):
        fmaps = torch.clone(feature_map_batch)
        fmaps.requires_grad = True
        fmaps.retain_grad()
        fmaps_logits = self.net.block(fmaps)
        loss = self.criterion_classi(fmaps_logits, target)
        loss.backward()
        saliency = fmaps.grad.data.detach().abs()
        return saliency

    def saliency_linear_weighting_enhance(self, feature_map_batch, saliency):
        normalized_saliency = saliency / saliency.cpu().numpy().flatten().mean()

        # element-wise multiplication
        enhanced_feature_map_batch = torch.mul(feature_map_batch, normalized_saliency)

        return enhanced_feature_map_batch

    def saliency_threshold_enhance(self, feature_map_batch, saliency):
        # TODO
        pass
        
    # update storage for reading attention and least memory usage indices based after each epoch
    def updateStorage_epoch(self, train_loader, run, task):
        self.net.evalModeOn()       
        print('=====================Storing replay examples=====================')
        avgSampleNum = math.floor(self.capacity/len(self.active_out_nodes))
        img_skip = math.floor(( ((len(train_loader)-1)*self.config["batch_size"]) / self.config["memory_size"])) # * (len(self.active_out_nodes) / self.config["n_class"]))
        print("IMG_SKIP:", img_skip)
        if img_skip < 1:
            img_skip = 1

        raw_feature_maps_save = []
        reconstr_feature_maps_save = []
        random_reconstr_feature_maps_save = []
        saliency_save = []
        enhanced_feature_maps_save = []
        target_save = []

        # Iterate over batches in train loader, storing a subset of examples
        for i, (inputs, target) in enumerate(train_loader):
        
            # transferring to gpu if applicable
            if self.gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            if self.config["storage_type"] == "enhanced_raw_feature":
                feature_maps = self.net.FeatureExtractor(inputs)
                reconstr_feature_maps = self.net.crumbr.reconstruct_fbank_top1(feature_maps)
                random_reconstr_feature_maps = self.net.random_crumbr.reconstruct_fbank_top1(feature_maps)
                saliency = self.get_saliency(feature_maps, target)
                enhanced_feature_maps = self.saliency_linear_weighting_enhance(feature_maps, saliency)
                stored_ex = enhanced_feature_maps.view(enhanced_feature_maps.size(0), -1).detach().cpu()

                self.optimizer_net.zero_grad()

                # Store all the tensors for saving
                raw_feature_maps_save.append(feature_maps.reshape(feature_maps.size(0), -1).detach().cpu().numpy())
                reconstr_feature_maps_save.append(reconstr_feature_maps.reshape(feature_maps.size(0), -1).detach().cpu().numpy())
                random_reconstr_feature_maps_save.append(random_reconstr_feature_maps.reshape(feature_maps.size(0), -1).detach().cpu().numpy())
                saliency_save.append(saliency.reshape(feature_maps.size(0), -1).detach().cpu().numpy())
                enhanced_feature_maps_save.append(enhanced_feature_maps.reshape(feature_maps.size(0), -1).detach().cpu().numpy())
                target_save.append(target.reshape(target.size(0), -1).detach().cpu().numpy())

            else:
                stored_ex = self.net.images_to_storage(inputs, storage_type=self.config["storage_type"])

            # Zhang, Baosheng, Yuchen Guo, Yipeng Li, Yuwei He, Haoqian Wang, and Qionghai Dai.
            # "Memory recall: A simple neural network training framework against catastrophic forgetting."
            # IEEE Transactions on Neural Networks and Learning Systems 33, no. 5 (2021): 2010-2022.
            if self.config["storage_type"] == "merec":
                for cls in self.active_out_nodes:

                    if cls not in self.memory_storage:
                        self.memory_storage[cls] = Welford()

                    class_examples_list = [stored_ex[i] for i in range(stored_ex.size(0)) if target[i] == cls]
                    for ex in class_examples_list:
                        self.memory_storage[cls].add(ex.numpy())

            else:
                all_classes_full = True  # Will be set to False if any class is not at capacity
                for cls in self.active_out_nodes:
                    all_examples_storage_list = [stored_ex[i] for i in range(stored_ex.size(0)) if target[i] == cls]
                    if len(all_examples_storage_list) > 0:
                        selected_examples_storage = torch.stack(all_examples_storage_list[0::img_skip], dim=0)

                        if cls in self.memory_storage:
                            self.memory_storage[cls] = torch.cat((self.memory_storage[cls], selected_examples_storage), dim=0)
                        else:
                            self.memory_storage[cls] = selected_examples_storage

                    # Pop out old examples (queue). Stored examples from old classes need to make space for new classes
                    if cls in self.memory_storage and self.memory_storage[cls].size(0) >= avgSampleNum:
                        # In "adaptive storage", shuffle the storage of each class to ensure iid removal of examples
                        perm = torch.randperm(self.memory_storage[cls].size(0))
                        self.memory_storage[cls] = self.memory_storage[cls][perm]
                        self.memory_storage[cls] = self.memory_storage[cls][(self.memory_storage[cls].size(0) - avgSampleNum):, :]
                    else:
                        all_classes_full = False

                if all_classes_full:
                    print("Stored sufficient examples for all classes")
                    break

        if self.config["storage_type"] == "enhanced_raw_feature":
            # Save feature tensors for analysis
            np.save(os.path.join(self.config["full_out_dir"], "raw_fm_r" + str(run) + "_t" + str(task)), np.vstack(raw_feature_maps_save))
            np.save(os.path.join(self.config["full_out_dir"], "reconstr_fm_r" + str(run) + "_t" + str(task)), np.vstack(reconstr_feature_maps_save))
            np.save(os.path.join(self.config["full_out_dir"], "random_reconstr_fm_r" + str(run) + "_t" + str(task)), np.vstack(random_reconstr_feature_maps_save))
            np.save(os.path.join(self.config["full_out_dir"], "saliency_fm_r" + str(run) + "_t" + str(task)), np.vstack(saliency_save))
            np.save(os.path.join(self.config["full_out_dir"], "enhanced_fm_r" + str(run) + "_t" + str(task)), np.vstack(enhanced_feature_maps_save))
            np.save(os.path.join(self.config["full_out_dir"], "target_r" + str(run) + "_t" + str(task)), np.vstack(target_save))

        if not self.config["storage_type"] == "merec":
            # Print memory storage stats
            for cls in self.active_out_nodes:
                print("Number of samples stored for class id " + str(cls) + ": " + str(self.memory_storage[cls].size(0)))
            sz_bytes = sum([sys.getsizeof(att_tensor.storage()) for att_tensor in self.memory_storage.values()])
            num_ex = sum([att_tensor.size(0) for att_tensor in self.memory_storage.values()])
            assert num_ex <= self.capacity, "Exceeded replay attention storage. Trying to store " + str(num_ex) + " examples with capacity=" + str(self.capacity)
            print("Size of replay attention storage for " + str(num_ex) + " examples in bytes: " + str(sz_bytes))

        if self.config["storage_type"] == "feature":
            # update least used memory slots
            self.used_mem_slots = torch.zeros(self.n_memblocks)
            for cls in range(self.config['n_class']):
                if cls in self.memory_storage.keys():
                    sz_w = self.memory_storage[cls][:, :-self.config['n_class']].size(0)
                    sz_h = self.memory_storage[cls][:, :-self.config['n_class']].size(1)
                    indices = torch.reshape(self.memory_storage[cls][:, :-self.config['n_class']], (sz_w, sz_h)).long()
                    self.used_mem_slots[indices] = 1
            print("Total number of memory slots used for replay: " + str(self.used_mem_slots.sum()))
            
        self.net.trainModeOn()

    # compute validation loss/accuracy (being called outside class)
    def validation(self, dataloader, metric_topk=(1,), test_perturbation=None, get_all_accs=False, ignore_partial_batches=False):

        accs_out_avg = {}
        accs_dir_avg = {}
        for k in metric_topk:
            accs_out_avg[k] = AverageMeter()
            accs_dir_avg[k] = AverageMeter()

        batch_timer = Timer()
        batch_timer.tic()
        
        self.net.evalModeOn()        
        # keeping track of prior mode
        if self.config['visualize']:
            self.viz_read_att = {i: [] for i in range(self.config['n_class'])}
            self.viz_input = {i: [] for i in range(self.config['n_class'])}
            self.viz_direct = {i: [] for i in range(self.config['n_class'])}
            self.confusemat = torch.zeros((len(self.active_out_nodes),len(self.active_out_nodes)))

        if get_all_accs:
            accs_out_all = []
            accs_dir_all = []
            targets_all = []
        
        for i, (inputs, target) in enumerate(dataloader):
            if self.gpu:
                with torch.no_grad():
                    inputs = inputs.cuda()
                    target = target.cuda()
             
            direct, output = self.predict(inputs, test_perturbation=test_perturbation)
            
            output = output[:,self.active_out_nodes]
            direct = direct[:,self.active_out_nodes]
            
            if self.gpu:
                target = target.cpu()

            accs_out = accuracy(output, target, metric_topk)
            accs_dir = accuracy(direct, target, metric_topk)
            if get_all_accs and len(target) == self.net.config["batch_size"]:
                accs_out_all.append(accs_out)
                accs_dir_all.append(accs_dir)
                targets_all.extend(target.tolist())

            for k_ind, k in enumerate(metric_topk):
                accs_out_avg[k].update(accs_out[k_ind], inputs.size(0))
                accs_dir_avg[k].update(accs_dir[k_ind], inputs.size(0))

        accs_out_avg = [acc_out.avg for acc_out in list(accs_out_avg.values())]
        accs_dir_avg = [acc_dir.avg for acc_dir in list(accs_dir_avg.values())]

        total_time = batch_timer.toc()
        if get_all_accs:
            return accs_out_avg, accs_dir_avg, accs_out_all, accs_dir_all, total_time, targets_all
        else:
            return accs_out_avg, accs_dir_avg, total_time

    # stream learning (being called outside class)
    def learn_stream(self, train_loader, run, task, epoch):
        
        self.net.trainOverallOn()
        losses_classidir = AverageMeter()
        losses_classiout = AverageMeter()
        losses = AverageMeter()
        losses_replay = AverageMeter()
        acc_out = AverageMeter()
        acc_dir = AverageMeter()
        
        data_timer = Timer()
        batch_timer = Timer()
        forward_timer = Timer()
        backward_timer = Timer()
        
        data_time = AverageMeter()
        batch_time = AverageMeter()
        forward_time = AverageMeter()
        backward_time = AverageMeter()
        
        self.log('Batch\t Loss\t\t Acc')
        data_timer.tic()
        batch_timer.tic()

        mem_stats = []
        
        # iterating over train loader
        for i, (inputs, target) in enumerate(train_loader):             
            # transferring to gpu if applicable
            if self.gpu:
                inputs = inputs.cuda()
                target = target.cuda()
            
            # measure data loading time
            data_time.update(data_timer.toc())           
            self.optimizer_net.zero_grad()

            # Record mem stats for the first and last batch
            if i == 0 or i == len(train_loader) - 1:
                get_mem_stats = True
            else:
                get_mem_stats = False

            # FORWARD pass
            # getting loss, updating model
            forward_timer.tic()

            direct_logits, output_logits, mem_stats_sample = self.net.forward(inputs, get_mem_stats=get_mem_stats, storage_type=self.config["storage_type"], use_random_resize_crops=self.config["use_random_resize_crops"])
            forward_time.update(forward_timer.toc())

            if mem_stats_sample is not None:
                mem_stats.append(mem_stats_sample)
                if task == 0 and epoch == 0 and i == 0:
                    print("Initial mem stats: " + str(mem_stats_sample))
            
            # BACKWARD pass
            backward_timer.tic()
            #classification loss
            loss_classiout = self.criterion_classi(output_logits, target)
            loss_classidir = self.criterion_classi(direct_logits, target)
            if self.config['pretraining'] or self.config['plus_direct_loss']:
                if self.config['pt_only_codebook_out_loss']:
                    # This is unablated CRUMB's loss during pretraining
                    loss = loss_classiout + loss_classidir
                else:
                    loss = loss_classiout + (0*loss_classidir)
            else:
                if (self.config["storage_type"] in ["image", "raw_feature", "enhanced_raw_feature"]) or self.config['direct_loss_only']:
                    loss = (0*loss_classiout) + loss_classidir
                else:
                    # This is unablated CRUMB's loss during stream learning
                    loss = loss_classiout + (0*loss_classidir)

            loss.backward()
            self.clip_grads(self.net)
            self.optimizer_net.step()                 
            ## IMPORTANT: have to detach memory for next step; otherwise, backprop errors
            #updateCRUMB
            backward_time.update(backward_timer.toc())

            if task > 0 or self.config['replay_in_1st_task']:
                ## REPLAY read content from classes not belonging to target labels from memory
                for time in range(self.config['replay_times']):
                    replay_size, replay_loss = self.criterion_replay(target, storage_type=self.config["storage_type"])
                if self.config['replay_times'] > 0 and replay_size > 0:
                    losses_replay.update(replay_loss, replay_size)
                else:
                    replay_size = 0
                    replay_loss = None
            
            # COMPUTE accuracy
            inputs = inputs.detach()
            target = target.detach()            
            # mask inactive output nodes
            direct_logits = direct_logits[:,self.active_out_nodes]
            output_logits = output_logits[:,self.active_out_nodes]

            acc_out.update(accuracy(output_logits, target, topk=self.config["acc_topk"])[0], inputs.size(0))
            acc_dir.update(accuracy(direct_logits, target, topk=self.config["acc_topk"])[0], inputs.size(0))
            losses.update(loss.detach().item(), inputs.size(0))
            losses_classiout.update(loss_classiout.detach().item(), inputs.size(0))
            losses_classidir.update(loss_classidir.detach().item(), inputs.size(0))
            
            # measure elapsed time for entire batch
            batch_time.update(batch_timer.toc())
            # updating these timers with with current time
            data_timer.toc()
            forward_timer.toc()
            backward_timer.toc()
            
            self.log('[{0}/{1}]\t'
                          'L = {loss.val:.3f} ({loss.avg:.3f})\t'
                          'L-dir = {losses_classidir.val:.3f} ({losses_classidir.avg:.3f})\t'
                          'L-out = {losses_classiout.val:.3f} ({losses_classiout.avg:.3f})\t'
                          'L-rep = {loss_replay.val:.3f} ({loss_replay.avg:.3f})\t'
                          'A-out = {acc_out.val:.2f} ({acc_out.avg:.2f})\t'
                          'A-dir = {acc_dir.val:.2f} ({acc_dir.avg:.2f})'.format(
                        i, len(train_loader), loss=losses, losses_classidir = losses_classidir, losses_classiout=losses_classiout, loss_replay=losses_replay, acc_out=acc_out, acc_dir=acc_dir))
            # self.log('{batch_time.val:3f}\t'
            #               '{data_time.val:3f}\t'
            #               '{forward_time.val:3f}\t {backward_time.val:3f}'.format(
            #             batch_time=batch_time, data_time=data_time, forward_time=forward_time, backward_time=backward_time))

        self.log(' * Train Acc: A-out = {acc_out.avg:.3f} \t A-dir = {acc_dir.avg:.3f}'.format(acc_out=acc_out, acc_dir=acc_dir))
        self.log(' * Avg. Data time: {data_time.avg:.3f}, Avg. Batch time: {batch_time.avg:.3f}, Avg. Forward time: {forward_time.avg:.3f}, Avg. Backward time: {backward_time.avg:.3f}'
                 .format(data_time=data_time, batch_time=batch_time, forward_time=forward_time, backward_time=backward_time))

        # END of epoch train data
        # UPDATE storage of reading attention
        # UPDATE least used memory slot indices
        if not self.config['pretraining']:  # no need for storing replay examples during pretraining
            if task > 0 or self.config['replay_in_1st_task'] or (task == 0 and epoch == self.config['n_epoch_first_task']-1):
                if not task == self.config['ntask']-1:  # no need for storing replay examples after the last task
                    self.updateStorage_epoch(train_loader, run, task)

        return mem_stats

    # overriding cuda function
    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.net = self.net.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        
        return self
    
    # save current model state
    def save_model(self, filename):
        net_state = self.net.state_dict()        
        netPath = filename + '_net.pth'        
        print('=> Saving models and aug memory to:', filename)
        torch.save(net_state, netPath)
        print('=> Save Done')
        
    def save_memory(self, filename, custom_memory=None):
        if custom_memory is None:
            net_state = self.net.crumbr.memory
        else:
            net_state = custom_memory
        netPath = filename + '_mem.pth'        
        print('=> Saving models and aug memory to:', filename)
        torch.save(net_state, netPath)
        print('=> Save Done')

    def clip_grads(self, model):
        """Gradient clipping to the range [10, 10]."""
        parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)

    def decrement_lr(self, lr_decrement, lr_min):
        if lr_decrement > 0:
            for g in self.optimizer_net.param_groups:
                if g['lr'] >= lr_min + lr_decrement:
                    g['lr'] = g['lr'] - lr_decrement
                else:
                    g['lr'] = lr_min
                print("Set lr to " + str(g['lr']))
        
    def visualize_att_read(self, filename):
        for i in self.active_out_nodes: #range(self.config['n_class']):
            att = self.viz_read_att[i]
            arr = np.vstack(att)   
            
            inputs = self.viz_input[i]
            arr_inputs = np.vstack(inputs)
            #sort in descending order and return top 5 index
            #print((-np.mean(arr, axis=0)).argsort(axis=0))
            
            direct = self.viz_direct[i]
            arr_direct = np.vstack(direct)
            
            sio.savemat(filename + '_att_read_' + str(i) + '.mat', {'att_read':arr,'inputs':arr_inputs, 'direct':arr_direct})
        sio.savemat(filename + '_confusemat.mat', {'confusemat': self.confusemat.numpy()})    
        
            #b = plt.imshow(arr[:self.viz_NumEgs,:], cmap='hot')
            #plt.colorbar(b)
            #plt.show()
        
    def visualize_memory(self, filename):
        plotmem = self.crumbr.net.memory.clone().detach().cpu()
        sio.savemat(filename + '_memory.mat', {'memory':plotmem.detach().cpu().numpy()})
        #b = plt.imshow(plotmem.detach().cpu().numpy(), cmap='hot')
        #plt.colorbar(b)
        #plt.show()
        