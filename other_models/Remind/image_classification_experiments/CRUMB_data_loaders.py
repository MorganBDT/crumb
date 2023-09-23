import torch
import torchvision.transforms as transforms
from dataloaders import datasets


def get_data_loader(split, min_class, max_class, args=None, return_item_ix=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if split == 'train':
        train_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            normalize,
        ])
        print('loading training set from run= ' + str(args.run) + '; task= ' + str(args.task))
        if args.dataset == 'core50':
            train_data = datasets.CORE50(
                dataroot=args.images_dir, filelist_root=args.filelist_root, scenario=args.scenario,
                offline=args.offline, run=args.run, batch=args.task, transform=train_transforms,
                returnIndex=return_item_ix)
        elif args.dataset in ['toybox', 'ilab2mlight', 'cifar100', 'ilab2mlight+core50', 'icubworldtransf']:
            train_data = datasets.Generic_Dataset(
                dataroot=args.images_dir, dataset=args.dataset, filelist_root=args.filelist_root,
                scenario=args.scenario,
                offline=args.offline, run=args.run, batch=args.task, transform=train_transforms,
                returnIndex=return_item_ix)
        else:
            raise ValueError("Invalid dataset name, try 'core50', 'toybox', or 'ilab2mlight' or 'cifar100'")

        # get train loader
        data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    elif split == 'val':
        test_transforms = transforms.Compose([
            transforms.Resize([224, 224]),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        if args.dataset == 'core50':
            test_data = datasets.CORE50(
                dataroot=args.images_dir, filelist_root=args.filelist_root, scenario=args.scenario,
                offline=args.offline, run=args.run, train=False, transform=test_transforms, returnIndex=return_item_ix)
        elif args.dataset in ['toybox', 'ilab2mlight', 'cifar100', 'ilab2mlight+core50', 'icubworldtransf']:
            # get test data
            test_data = datasets.Generic_Dataset(
                dataroot=args.images_dir, dataset=args.dataset, filelist_root=args.filelist_root,
                scenario=args.scenario, offline=args.offline,
                run=args.run, train=False, transform=test_transforms, returnIndex=return_item_ix)
        else:
            raise ValueError("Invalid dataset name, try 'core50' or 'toybox' or 'ilab2mlight' or 'cifar100'")

        task_inds = list(range(min_class, max_class))
        print('check me .......... selected classes: ', task_inds)
        test_inds = [i for i in range(len(test_data)) if
                     test_data.labels[i] in task_inds]  # list(range(len(test_data)))

        task_test_data = torch.utils.data.Subset(test_data, test_inds)
        data_loader = torch.utils.data.DataLoader(
            task_test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True)

    else:
        raise ValueError("'!!!!!!!!!! val or train; nothing else !!!!")

    #     data_loader = utils_imagenet.get_imagenet_data_loader(images_dir + '/' + split, label_dir, split,
    #                                                           batch_size=batch_size, shuffle=False, min_class=min_class,
    #                                                           max_class=max_class, return_item_ix=return_item_ix)
    return data_loader