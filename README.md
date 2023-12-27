# Compositional Replay Using Memory Blocks (CRUMB)

From the paper "[Tuned Compositional Feature Replays for Efficient Stream Learning](http://klab.tch.harvard.edu/publications/PDFs/gk8019.pdf)."
Authors: Morgan Talbot*, Rushikesh Zawar*, Rohil Badkundri, Mengmi Zhang†, and Gabriel Kreiman† (* equal contribution, † corresponding authors)

## Project description 

Stream learning refers to the ability to acquire and transfer knowledge across a continuous stream temporally correlated  data without forgetting and without repeated passes over the data. A common way to avoid catastrophic forgetting is to intersperse new examples with replays of old examples stored as image pixels or reproduced by generative models. We propose a new continual learning algorithm, Compositional Replay Using Memory Blocks (CRUMB), which mitigates forgetting by replaying feature maps reconstructed by recombining generic parts. CRUMB concatenates trainable and re-usable "memory block" vectors to compositionally reconstruct feature map tensors in convolutional neural networks, like crumbs forming a loaf of bread. CRUMB stores the indices of memory blocks used to reconstruct new stimuli, enabling replay of specific memories during later tasks. This reconstruction mechanism also primes the neural network to minimize catastrophic forgetting by forcing it to attend to information about object shapes more than information about image textures, and stabilizes the network during stream learning by providing a shared feature-level basis for all training examples. These properties allow CRUMB to outperform an otherwise identical algorithm that stores and replays raw images while occupying only 3.6% as much memory. We stress-tested CRUMB alongside 13 competing methods on 7 challenging datasets. To address the limited number of existing online stream learning datasets, we introduce 2 new benchmarks (Toybox and iLab) by adapting existing datasets for stream learning. With about 4% as much memory and 30% as much runtime, CRUMB mitigates catastrophic forgetting more effectively than the prior state of the art.

## Setup

This PYTORCH project was developed and tested using Ubuntu version 20.04, CUDA 10.1, and Python version 3.6. See ```requirements.txt``` for package versions. Additional requirements: ffmpeg

Refer to [link](https://www.anaconda.com/distribution/) for Anaconda installation. Alternatively, execute the following command:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh
```
After Anaconda installation, create a conda environment (here, our conda environment is called "crumb"):
```
conda create -n crumb python=3.6
```
Activate the conda environment:
```
conda activate crumb
```
In the conda environment, 
```
pip install -r requirements.txt
```
Install ffmpeg (the final command is to verify installation):
```
sudo apt update
sudo apt install ffmpeg
ffmpeg -version
```
Finally, clone this repository from GitHub. 

## Preparing/indexing the datasets

This project uses three video datasets: CORe50, Toybox, and iLab-2M-Light. Each dataset has its own procedure for pre-processing and indexing (described below). In general, this involves the automated generation of a "dirmap" csv file for each dataset that indexes all of the images in the dataset - this dirmap is used to create train/test splits and select sequences/batches of images for training and testing under each of the 2 paradigms described in the paper (class_iid, class_instance) - this information will be generated in directories (one per dataset) named "<datasetname>_task_filelists. Although each dataset has its own indexing procedure to produce <datasetname>_dirmap.csv, the same functions are used to process this csv and produce <datasetname>_task_filelists. The task_filelists are then used for training and testing by the shell scripts for each agent (i.e. CRUMB and a variety of baseline agents like EWC, iCARL, REMIND) found in the "scripts" folder.

### Core50 dataset

1. Download the Core50 dataset from [this page](https://vlomonaco.github.io/core50/).
2. Unlike with the other two datasets below, the pre-generated core50_dirmap.csv provided with this repo can be used out-of-the-box. From the root project directory, run "sh scripts/setup_tasks_core50.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "core50_task_filelists"

### Toybox dataset

1. Download all three parts of the Toybox dataset from [this page](https://aivaslab.github.io/toybox/). Create a Toybox dataset directory called "toybox" (or whatever you like) and extract all three of the downloaded dataset components into this directory. Your Toybox dataset directory should contain the directories "animals", "vehicles", and "households"
2. Move the pre-generated toybox_dirmap.csv out of the "dataloaders" directory to a safe place. You still need to generate a new copy of this file yourself (see below), but a pre-generated copy is provided so that you can see what it's supposed to look like. 
3. Run the toybox_dirmap.py script to extract frames from the Toybox dataset videos and generate the dirmap csv to index them.  This script requires ffmpeg, which it uses to extract images from the videos in the Toybox dataset at a rate of 1 fps. This script will take several hours to run. Navigate to the "dataloaders" folder in the command line, and run "python toybox_dirmap.py <dataset_path>", replacing <dataset_path> with the location of your Toybox dataset directory. There should now be a "toybox_dirmap_unbalanced.csv" in the "dataloaders" folder, and in your Toybox dataset directory there should now be a new directory called "images"
4. Run the toybox_sample.py script to sample a slightly reduced version of the dataset with balanced statistics (i.e. guaranteed exactly the same number of objects per class, images per object, etc). This script may take several minutes to run. Navigate to the "dataloaders" directory, and run "python toybox_sample.py". There should now be a "toybox_dirmap.csv" in the "dataloaders" folder.
5. (Recommended) Verify that the dataset is correctly balanced by navigating to the "dataloaders" directory and running "python dirmap_csv_stats.py toybox_dirmap.csv". You should see that there are exactly 4350 examples (images) in each of the 12 classes, that each session has exactly 15 images (avg/min/max all equal 15), that each object has exactly 10 sessions, and that each class has exactly 29 objects. 
6. From the root project directory, run "sh scripts/setup_tasks_toybox.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "toybox_task_filelists"

### iLab-2M-Light dataset

1. Download the iLab-2M-Light dataset from [this page](https://bmobear.github.io/projects/viva/). Direct download link [HERE](http://ilab.usc.edu/ilab2m/iLab-2M-Light.tar.gz). Extract the dataset into a directory of your choice.
2. Move the pre-generated ilab2mlight_dirmap.csv out of the "dataloaders" directory to a safe place. You still need to generate a new copy of this file yourself (see below), but a pre-generated copy is provided so that you can see what it's supposed to look like. 
3. Run the ilab2mlight_dirmap.py script to generate the dirmap csv indexing the redistributed images in the dataset. This script should only take a few minutes to run. Navigate to the "dataloaders" folder in the command line, and run "python ilab2mlight_dirmap.py <dataset_path>", replacing <dataset_path> with the location of your iLab dataset directory. There should now be an "ilab2mlight_dirmap_all.csv" in the "dataloaders" folder
4. Run the ilab2mlight_sample.py script to sample a slightly reduced version of the dataset with balanced statistics (i.e. guaranteed exactly the same number of objects per class, images per object, etc). This script may take several minutes to run. Navigate to the "dataloaders" directory, and run "python ilab2mlight_sample.py". There should now be an "ilab2mlight_dirmap_massed.csv" in the "dataloaders" folder.
5. (Recommended) Verify that the dataset is correctly balanced by navigating to the "dataloaders" directory and running "python dirmap_csv_stats.py ilab2mlight_dirmap.csv". You should see that there are exactly 3360 examples (images) in each of the 14 classes, that each session has exactly 15 images (avg/min/max all equal 15), that each object has exactly 8 sessions, and that each class has exactly 28 objects. 
6. Distribute the images in the dataset to a nested directory structure by running the ilab2mlight_distribute_img_dirs.py script. The dataset comes by default with all of the images massed together in one directory, and this can make loading the data very slow during training. Navigate to the "dataloaders" folder and run "python ilab2mlight_distribute_img_dirs.py <dataset_path> <distributed_dataset_path>". The <distributed_dataset_path> should be a path to a new directory in which the distributed version of the dataset will be placed. Make sure you have enough room on your HDD/SSD before running this script, as it will make a copy of all of the sampled iamges in the dataset. This script will take several hours to run (e.g. maybe 12 hours). When it's finished, you should have "ilab2mlight_dirmap.py" in the "dataloaders" directory.
7. From the root project directory, run "sh scripts/setup_tasks_ilab2mlight.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "ilab2mlight_task_filelists"

### iCubWorld Transformations dataset

1. Download the iCubWorld Transformations dataset from [this page](https://robotology.github.io/iCubWorld/#icubworld-transformations-modal). 
2. Extract everything into a dataset called "icubworldtransf" that contains directories "part1", "part2", "part3", and "part4". 
3. Run the icub python script as "python dataloaders/icubworldtransf_dirmap.py --dataset_root my/path/to/icubworldtransf"
4. OPTIONAL: To save space on your computer, run "python dataloaders/icubworldtransf_sparse.py" (be sure to change the definition of variable "dataset_root_dir" in that script first). It will make a new dataset directory alongside the first one with only the subset of images you need for our benchmark (which is much fewer than in the entire original dataset, which has a high frame rate and both left and right cams)
5. From the root project directory, run "sh scripts/setup_tasks_icubworldtransf.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "icubworldtransf_task_filelists"

### iLab+CORe50 dataset

1. From the root project directory, run "sh scripts/setup_tasks_ilab2mlight+core50.sh". This should only take a few minutes. You should now see a new folder in the "dataloaders" directory called "ilab2mlight+core50_task_filelists"

## Running CRUMB

First, conduct additional pretraining of CRUMB for 10 epochs on ImageNet by running the script:
```
./scripts/crumb_pretrain.sh imagenet 0 0 0.001 256 16 0 "imagenet/location"
```
(See further instructions in scripts/crumb_pretrain.sh)

A new results folder for PRETRAINING, such as "crumb_imagenet_pretrain_run0.sh" should appear in the root directory of the repository with .csv files and a saved copy of the pretrained model.

Then, for stream learning, run the script:
```
./scripts/crumb_stream.sh core50 0 "_MyExperiment" 0.001 0 "./my_pretrain_dir" 256 16 1000 "dataset/location"
```
where core50 can be replaced by toybox, ilab2mlight, cifar100, or imagenet, "my_pretrain_dir" should be set to the location of the top-level directory created during pretraining, and "dataset/location" should be set to the dataset location. 

A new results folder for STREAM LEARNING should appear in the root directory of the repository with .csv files and saved versions of the model. 

The instructions above will complete 10 runs of stream learning (5 for imagenet) based off of a single pretraining run on imagenet. For 5 stream learning runs using 5 independent pretraining runs, see instructions in ./scripts/crumb_stream_1run_per_pt.sh. 

The file that contains the accuracy used for the paper will be called "top1_test_all_direct_all_runs.csv". If you average the final column, you should get results similar to entries from the main table in the paper. 
The naming convention for the .csv files is as follows with variable parts indicated by < >: 
```
top<k>_test_<tasks_included>_<model_feature_source>_all_runs.csv"
```
Where k is 1 or 5 for top-1 or top-5 accuracy (note - top-5 accuracy is an almost useless metric for CORe50, Toybox, and iLab which all have <=14 classes),
tasks_included is "1st" for accuracy on the classes in the first task only, "current" for accuracy on classes from the current task only, "all" for accuracy on classes from all previously learned tasks (including the current task). "all" is what we use throughout the paper. 
and model_feature_source is "direct" for all results used in this paper. "mem" is what the accuracy would have been if we used the prediction arising from the reconstructed feature bank, which tends to be less accurate - when processing a new image (but not a stored replay), the original feature bank is available to us.

The all_epochs folder contains accuracies for all training epochs - e.g. if we trained on the first task for multiple epochs, the model's accuracy after each epoch can be found in a .csv in this folder. 

## Running baseline algorithms
Make sure that your current working directory is at the root of the repository. Run the following (for example) to train and save results of individual algorithm on CoRE50 dataset:
```
./scripts/optimal_core50/EWC.sh core50 0
```
***NOTE*** Set ```DATAROOT``` in each shell script to the directory where the dataset is downloaded and stored.
Each shell script is an algorithm. It takes two input arguments: 
 - the dataset names: core50, toybox, ilab2mlight
 - the GPU ID to run the jobs: e.g. 0
```
Once the algorithms finish running, .csv files with accuracy results after training on each task (on first task, the current task, and all learned tasks) will be saved to a new directory created in the top-level repository folder. 
```
Similarly, one can run shell scripts stored in ```optimal_toybox```, ```optimal_ilab```, ```optimal_cifar100```, and ```optimal_imagenet``` folders.
A number of baseline algorithms (e.g., REMIND, COPE, GSS, Rainbow Memory) are not included in this unified framework, and are found in other_models, with various procedures required to run them on the datasets. 

## Running ablation study/"model analysis" experiments
Run shell scripts in the ```scripts/ablation``` folder for model analysis experiments. Some ablations (e.g., changing the size of the codebook) require specifically ablated pretraining on ImageNet followed by stream learning, while others (e.g., freezing the codebook during streaming) can all rely on unablated pretraining runs. **You will likely need to modify the OUTDIR and/or PRETRAIN_DIR in the scripts in order to have the proper pretraining directory successfully loaded for stream learning, and also to avoid overwriting previous stream learning runs (e.g. crumb_stream_1run_per_pt.sh is used for many of the ablations, but by default it always saves to a directory of the same name... specify a different SUFFIX as a command line argument).** 

Scripts/ablation/ablation_all_batches.sh followed by network_analysis_stats.py can be used to run statistical tests on pairwise comparisons between CRUMB and each perturbed version of it. ablation_all_batches.sh gets the accuracy on each batch of (default) 100 images in the test set, and then network_analysis_stats.py uses this data to run statistical comparisons. 

This is the correspondence between the model analysis arms in the paper (some may be in supplementary materials) and the scripts: 
* **Ours**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt.sh
* **Image replay**: (no CRUMB pretraining) -> crumb_stream_1run_per_no_pt_image_replay.sh
* **Ours p.t. + im. rep.**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_image_replay.sh
* **Early feature replay**: crumb_pretrain_imagenet_cutlayer_3.sh -> crumb_stream_1run_per_pt_cutlayer_3.sh
* **MeRec replay**: (no CRUMB pretraining) -> crumb_stream_1run_per_no_pt_merec.sh
* **Half capacity**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_half_mem_cap.sh
* **Quarter capacity**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_quarter_mem_cap.sh
* **No replay**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_no_replay.sh
* **Vanilla pretrain**: (no CRUMB pretraining) -> crumb_stream_1run_per_pt.sh
* **Pretrain weights**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_weights_only.sh
* **Pretrain mem. blocks**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_memblocks_only.sh
* **CIFAR100 pretrain**: crumb_pretrain_cifar100.sh -> crumb_stream_1run_per_pt.sh
* **Freeze memory**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_freeze_memory.sh
* **Normal init.**: crumb_pretrain_imagenet_stdnml.sh -> crumb_stream_1run_per_pt.sh
* **Uniform init.**: crumb_pretrain_imagenet_uniform.sh -> crumb_stream_1run_per_pt.sh
* **Dense matched init.**: crumb_pretrain_imagenet_distmatch_dense.sh -> crumb_stream_1run_per_pt.sh
* **Different number of memory blocks**: crumb_pretrain_imagenet_n_codebook_rows.sh (with n replaced by the number of blocks, must be power of 2 <= 512 or add another script..) -> crumb_stream_1run_per_pt.sh (but specify the codebook size as a command line argument, like ./scripts/ablation crumb_stream_1run_per_pt.sh core50 0 <pretraining_dir> n_memory_blocks 1000 **n**)
* **Different memory block size**: crumb_pretrain_imagenet_n_codebook_feat.sh -> crumb_stream_1run_per_pt.sh (but specify the codebook size as a command line argument, like ./scripts/ablation crumb_stream_1run_per_pt.sh core50 0 <pretraining_dir> n_memory_blocks 1000 256 **n**)
* **Ours - direct loss**: crumb_pretrain_imagenet_no_direct_loss.sh -> crumb_stream_1run_per_pt.sh
* **Ours + direct loss**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_plus_direct_loss.sh
* **Direct loss**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_direct_loss_only.sh
* **MobileNetV2 CNN**: crumb_pretrain_imagenet_MobileNet_normal.sh -> crumb_stream_1run_per_pt_MobileNet.sh

For **Ours** vs **Image replay** on datasets other than CORe50, use the same scripts as above but specify a different dataset as the first command line argument to the script. For **Ours** vs **Image replay** in CORe50 with varying memory capacities (e.g., here with 6400 total images that can be stored, more than the size of the whole dataset):
* **Ours**: crumb_pretrain_imagenet_unablated.sh -> crumb_stream_1run_per_pt_var_mem_cap.sh core50 0 6400
* **Image replay**: (no CRUMB pretraining) -> crumb_stream_1run_per_no_pt_image_replay_var_mem_cap.sh core50 0 6400

## License
See [Kreiman lab](http://klab.tch.harvard.edu/code/license_agreement.pdf) for license agreements before downloading and using our source codes and datasets. The source code is intended for illustration purposes only. We do not provide technical support, but we would be happy to discuss science! 

