import os
import time
import copy
import torch
import argparse
import random
import numpy as np
from models.segmentation.segmentation import deeplab_v2_contrastive
from torch.utils.tensorboard import SummaryWriter
from utils.common import colors_city, colors_voc, categories_voc, categories_city, sizes_city, sizes_voc, \
    num_classes_voc, num_classes_city, coco_mean, coco_std, imagenet_mean, imagenet_std, \
    load_checkpoint, generate_class_balanced_pseudo_labels
from utils.losses import DynamicMutualLoss
from utils.module_list import *
from train import *





def after_loading():
    global lr_scheduler

    # The "poly" policy, variable names are confusing(May need reimplementation)
    if not args.labeling:
        if args.state == 2:
            len_loader = (len(labeled_loader) * args.epochs)
        else:
            len_loader = (len(pseudo_labeled_loader) * args.epochs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / len_loader) ** 0.9)

    # Resume training?
    if args.continue_from is not None:
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='PyTorch 1.6.0 && torchvision 0.7.0')
    parser.add_argument('--exp-name', type=str, default='auto',
                        help='Name of the experiment (default: auto)')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Train/Evaluate on PASCAL VOC 2012(voc)/Cityscapes(city) (default: voc)')
    parser.add_argument('--gamma1', type=float, default=0,
                        help='Gamma for entropy minimization in agreement (default: 0)')
    parser.add_argument('--gamma2', type=float, default=0,
                        help='Gamma for learning in disagreement (default: 0)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--val-num-steps', type=int, default=500,
                        help='How many steps between validations (default: 500)')
    parser.add_argument('--label-ratio', type=float, default=0.2,
                        help='Initial labeling ratio (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate (default: 0.002)')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay for SGD (default: 0.0005)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs for the fully-supervised initialization (default: 30)')
    parser.add_argument('--batch-size-labeled', type=int, default=1,
                        help='Batch size for labeled data (default: 1)')
    parser.add_argument('--batch-size-pseudo', type=int, default=7,
                        help='Batch size for pseudo labeled data (default: 7)')
    parser.add_argument('--do-not-save', action='store_false', default=True,
                        help='Save model (default: True)')
    parser.add_argument('--coco', action='store_true', default=False,
                        help='Models started from COCO in Caffe(True) or ImageNet in Pytorch(False) (default: False)')
    parser.add_argument('--valtiny', action='store_true', default=False,
                        help='Use valtiny instead of val (default: False)')
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Turn off data augmentations for pseudo labeled data (default: False)')
    parser.add_argument('--mixed-precision', action='store_true', default=False,
                        help='Enable mixed precision training (default: False)')
    parser.add_argument('--labeling', action='store_true', default=False,
                        help='Just pseudo labeling (default: False)')
    parser.add_argument('--continue-from', type=str, default=None,
                        help='Self-training begins from a previous checkpoint/Test on this')
    parser.add_argument('--train-set', type=str, default='1',
                        help='e.g. 1:7(8), 1:3(4), 1:1(2), 1:0(1) labeled/unlabeled split (default: 1)')
    parser.add_argument('--sets-id', type=int, default=0,
                        help='Different random splits(0/1/2) (default: 0)')
    parser.add_argument('--state', type=int, default=1,
                        help="Final test(3)/Fully-supervised training(2)/Semi-supervised training(1)")
    parser.add_argument('--sup_contrastive', action='store_true')
    parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
    parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
    parser.add_argument('--accum_iter', default=1, type=int, help='number of batches gradient accumalation')
    parser.add_argument('--strong_threshold', default=0.97, type=float)
    parser.add_argument('--temp', default=0.5, type=float)
    args = parser.parse_args()

    # Basic configurations
    exp_name = str(int(time.time()))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.exp_name != 'auto':
        exp_name = args.exp_name

    with open('config/' + exp_name + '_config.txt', 'w') as f:
        f.write(str(vars(args)))

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # accelerator = Accelerator(split_batches=True)
    # device = accelerator.device
    if args.coco:  # This Caffe pre-trained model takes "inhuman" mean/std & input format
        mean = coco_mean
        std = coco_std
        keep_scale = True
        reverse_channels = True
    else:
        mean = imagenet_mean
        std = imagenet_std
        keep_scale = False
        reverse_channels = False
    if args.dataset == 'voc':
        num_classes = num_classes_voc
        input_sizes = sizes_voc
        categories = categories_voc
        colors = colors_voc
    elif args.dataset == 'city':
        num_classes = num_classes_city
        input_sizes = sizes_city
        categories = categories_city
        colors = colors_city
    else:
        raise ValueError

    # net = deeplab_v2(num_classes=num_classes)
    net = deeplab_v2_contrastive(num_classes=num_classes)
    print(device)
    net.to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Just to be safe (a little bit more memory, by all means, save it to disk if you want)
    if args.state == 1:
        st_optimizer_init = copy.deepcopy(optimizer.state_dict())

    # Testing
    if args.state == 3:
        # net, optimizer = accelerator.prepare(net, optimizer)
        test_loader = init(batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                           state=3, split=None, valtiny=args.valtiny, no_aug=args.no_aug,
                           input_sizes=input_sizes, data_set=args.dataset, sets_id=args.sets_id,
                           mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
        load_checkpoint(net=net, optimizer=None, lr_scheduler=None,
                        is_mixed_precision=args.mixed_precision, filename=args.continue_from)
        test_one_set(loader=test_loader, device=device, net=net, categories=categories, num_classes=num_classes,
                     output_size=input_sizes[2], is_mixed_precision=args.mixed_precision)
    else:
        x = 0
        criterion = DynamicMutualLoss(gamma1=args.gamma1, gamma2=args.gamma2, ignore_index=255)
        writer = SummaryWriter('logs/' + exp_name)

        # Only fully-supervised training
        if args.state == 2:
            labeled_loader, val_loader = init(batch_size_labeled=args.batch_size_labeled,
                                              batch_size_pseudo=args.batch_size_pseudo, sets_id=args.sets_id,
                                              valtiny=args.valtiny,
                                              state=2, split=args.train_set, input_sizes=input_sizes,
                                              data_set=args.dataset,
                                              mean=mean, std=std, keep_scale=keep_scale, no_aug=args.no_aug,
                                              reverse_channels=reverse_channels)
            after_loading()
            # net, optimizer, labeled_loader = accelerator.prepare(net, optimizer, labeled_loader)
            x = train(exp_name=args.exp_name, writer=writer, loader_c=labeled_loader, loader_sup=None, validation_loader=val_loader,
                      device=device, criterion=criterion, net=net, optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      num_epochs=args.epochs, categories=categories, num_classes=num_classes,
                      is_mixed_precision=args.mixed_precision, with_sup=False, sup_contrative = args.sup_contrastive,
                      accum_iter=args.accum_iter, strong_threshold=args.strong_threshold, 
                      temp=args.temp, num_queries=args.num_queries, num_negatives=args.num_negatives,
                      val_num_steps=args.val_num_steps, input_sizes=input_sizes)

        # Self-training
        elif args.state == 1:
            if args.labeling:
                unlabeled_loader = init(
                    valtiny=args.valtiny, no_aug=args.no_aug, data_set=args.dataset,
                    batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                    state=0, split=args.train_set, input_sizes=input_sizes,
                    sets_id=args.sets_id, mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
                after_loading()
                # net, optimizer = accelerator.prepare(net, optimizer)
                time_now = time.time()
                ratio = generate_class_balanced_pseudo_labels(net=net, device=device, loader=unlabeled_loader,
                                                              input_size=input_sizes[2],
                                                              label_ratio=args.label_ratio, num_classes=num_classes,
                                                              is_mixed_precision=args.mixed_precision)
                print(ratio)
                print('Pseudo labeling time: %.2fs' % (time.time() - time_now))
            else:
                labeled_loader, pseudo_labeled_loader, val_loader = init(
                    valtiny=args.valtiny, no_aug=args.no_aug, data_set=args.dataset,
                    batch_size_labeled=args.batch_size_labeled, batch_size_pseudo=args.batch_size_pseudo,
                    state=1, split=args.train_set, input_sizes=input_sizes,
                    sets_id=args.sets_id, mean=mean, std=std, keep_scale=keep_scale, reverse_channels=reverse_channels)
                after_loading()

                x = train(exp_name=args.exp_name,writer=writer, loader_c=pseudo_labeled_loader, loader_sup=labeled_loader,
                          validation_loader=val_loader, lr_scheduler=lr_scheduler,
                          device=device, criterion=criterion, net=net, optimizer=optimizer,
                          num_epochs=args.epochs, categories=categories, num_classes=num_classes,
                          is_mixed_precision=args.mixed_precision, with_sup=True, sup_contrative = args.sup_contrastive,
                          accum_iter=args.accum_iter, strong_threshold=args.strong_threshold, 
                          temp=args.temp, num_queries=args.num_queries, num_negatives=args.num_negatives,
                          val_num_steps=args.val_num_steps, input_sizes=input_sizes)
                
                

        else:
            # Support unsupervised learning here if that's what you want
            # But we do not think that works, yet...
            raise ValueError

        if not args.labeling:
            # --do-not-save => args.do_not_save = False
            if args.do_not_save:  # Rename the checkpoint
                os.rename('temp.pt', exp_name + '.pt')
            else:  # Since the checkpoint is already saved, it should be deleted
                os.remove('temp.pt')

            writer.close()

            with open('log.txt', 'a') as f:
                f.write(exp_name + ': ' + str(x) + '\n')
