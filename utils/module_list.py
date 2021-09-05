import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

import torchvision.transforms.functional as transforms_f
from utils.common import ConfusionMatrix
from utils.datasets import StandardSegmentationDataset
from utils.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomCrop, Resize, LabelMap, \
    ZeroPad, Compose, RandomScale
from utils.common import base_city, base_voc, label_id_map_city



# --------------------------------------------------------------------------------
# Define ReCo loss
# --------------------------------------------------------------------------------
def contrastive_loss(rep, label, mask, prob, strong_threshold=1.0, temp=0.5, num_queries=256, num_negatives=256):
    batch_size, num_feat, im_w_, im_h = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    # compute valid binary mask for each pixel
    valid_pixel = label * mask

    # permute representation for indexing: batch x im_h x im_w x feature_channel
    rep = rep.permute(0, 2, 3, 1)

    # compute prototype (class mean representation) for each class across all valid pixels
    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []
    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]  # select binary mask for i-th class
        if valid_pixel_seg.sum() == 0:  # not all classes would be available in a mini-batch
            continue

        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()  # select hard queries

        seg_proto_list.append(torch.mean(rep[valid_pixel_seg.bool()], dim=0, keepdim=True))
        seg_feat_all_list.append(rep[valid_pixel_seg.bool()])
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    # compute regional contrastive loss
    if len(seg_num_list) <= 1:  # in some rare cases, a small mini-batch might only contain 1 or no semantic class
        return torch.tensor(0.0)
    else:
        reco_loss = torch.tensor(0.0)
        seg_proto = torch.cat(seg_proto_list)
        valid_seg = len(seg_num_list)
        seg_len = torch.arange(valid_seg)

        for i in range(valid_seg):
            # sample hard queries
            if len(seg_feat_hard_list[i]) > 0:
                seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,))
                anchor_feat_hard = seg_feat_hard_list[i][seg_hard_idx]
                anchor_feat = anchor_feat_hard
            else:  # in some rare cases, all queries in the current query class are easy
                continue

            # apply negative key sampling (with no gradients)
            with torch.no_grad():
                # generate index mask for the current query class; e.g. [0, 1, 2] -> [1, 2, 0] -> [2, 0, 1]
                seg_mask = torch.cat(([seg_len[i:], seg_len[:i]]))

                # compute similarity for each negative segment prototype (semantic class relation graph)
                proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
                proto_prob = torch.softmax(proto_sim / temp, dim=0)

                # sampling negative keys based on the generated distribution [num_queries x num_negatives]
                negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
                samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
                samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

                # sample negative indices from each negative class
                negative_num_list = seg_num_list[i+1:] + seg_num_list[:i]
                negative_index = negative_index_sampler(samp_num, negative_num_list)

                # index negative keys (from other classes)
                negative_feat_all = torch.cat(seg_feat_all_list[i+1:] + seg_feat_all_list[:i])
                negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

                # combine positive and negative keys: keys = [positive key | negative keys] with 1 + num_negative dim
                positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
                all_feat = torch.cat((positive_feat, negative_feat), dim=1)

            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
            reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, torch.zeros(num_queries).long().to(device))
        return reco_loss / valid_seg


def negative_index_sampler(samp_num, seg_num_list):
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            negative_index += np.random.randint(low=sum(seg_num_list[:j]),
                                                high=sum(seg_num_list[:j+1]),
                                                size=int(samp_num[i, j])).tolist()
    return negative_index



# --------------------------------------------------------------------------------
# Define useful functions
# --------------------------------------------------------------------------------
def label_binariser(inputs):
    outputs = torch.zeros_like(inputs).to(inputs.device)
    index = torch.max(inputs, dim=1)[1]
    outputs = outputs.scatter_(1, index.unsqueeze(1), 1.0)
    return outputs


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    # remap invalid pixels (-1) into 0, otherwise we cannot create one-hot vector with negative labels.
    # we will still mask out those invalid values in valid mask
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w]).to(inputs.device)
    aa = inputs.unsqueeze(1)
    aa[aa == 255] = 0
    bb = outputs.scatter_(1, aa, 1.0)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)


def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2


def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def tensor_to_pil(im, label, logits):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits

def init(batch_size_labeled, batch_size_pseudo, state, split, input_sizes, sets_id, std, mean,
         keep_scale, reverse_channels, data_set, valtiny, no_aug):
    # Return data_loaders/data_loader
    # depending on whether the state is
    # 0: Pseudo labeling
    # 1: Semi-supervised training
    # 2: Fully-supervised training
    # 3: Just testing

    # For labeled set divisions
    split_u = split.replace('-r', '')
    split_u = split_u.replace('-l', '')

    
    # Transformations (compatible with unlabeled data/pseudo labeled data)
    # ! Can't use torchvision.Transforms.Compose
    if data_set == 'voc':
        base = base_voc
        workers = 4
        transform_train = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
             RandomScale(min_scale=0.5, max_scale=1.5),
             RandomCrop(size=input_sizes[0]),
             RandomHorizontalFlip(flip_prob=0.5),
             Normalize(mean=mean, std=std)])
        if no_aug:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
                 Normalize(mean=mean, std=std)])
        else:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 RandomScale(min_scale=0.5, max_scale=1.5),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
        # transform_pseudo = Compose(
        #     [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
        #      Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
        #      Normalize(mean=mean, std=std)])
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             ZeroPad(size=input_sizes[2]),
             Normalize(mean=mean, std=std)])
    elif data_set == 'city':  # All the same size (whole set is down-sampled by 2)
        base = base_city
        workers = 8
        transform_train = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
             Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
             RandomScale(min_scale=0.5, max_scale=1.5),
             RandomCrop(size=input_sizes[0]),
             RandomHorizontalFlip(flip_prob=0.5),
             Normalize(mean=mean, std=std),
             LabelMap(label_id_map_city)])
        if no_aug:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
                 Normalize(mean=mean, std=std)])
        else:
            transform_train_pseudo = Compose(
                [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
                 # RandomResize(min_size=input_sizes[0], max_size=input_sizes[1]),
                 Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
                 RandomScale(min_scale=0.5, max_scale=1.5),
                 RandomCrop(size=input_sizes[0]),
                 RandomHorizontalFlip(flip_prob=0.5),
                 Normalize(mean=mean, std=std)])
        # transform_pseudo = Compose(
        #     [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
        #      Resize(size_image=input_sizes[0], size_label=input_sizes[0]),
        #      Normalize(mean=mean, std=std),
        #      LabelMap(label_id_map_city)])
        transform_test = Compose(
            [ToTensor(keep_scale=keep_scale, reverse_channels=reverse_channels),
             Resize(size_image=input_sizes[2], size_label=input_sizes[2]),
             Normalize(mean=mean, std=std),
             LabelMap(label_id_map_city)])
    else:
        base = ''

    # Not the actual test set (i.e.validation set)
    test_set = StandardSegmentationDataset(root=base, image_set='valtiny' if valtiny else 'val',
                                           transforms=transform_test, label_state=0, data_set=data_set)
    val_loader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=batch_size_labeled + batch_size_pseudo,
                                             num_workers=workers, shuffle=False)

    # Testing
    if state == 3:
        return val_loader
    else:
        # Fully-supervised training
        if state == 2:
            labeled_set = StandardSegmentationDataset(root=base, image_set=(str(split) + '_labeled_' + str(sets_id)),
                                                      transforms=transform_train, label_state=0, data_set=data_set)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set, batch_size=batch_size_labeled,
                                                         num_workers=workers, shuffle=True)
            return labeled_loader, val_loader

        # Semi-supervised training
        elif state == 1:
            pseudo_labeled_set = StandardSegmentationDataset(root=base, data_set=data_set, mask_type='.npy',
                                                             image_set=(str(split_u) + '_unlabeled_' + str(sets_id)),
                                                             transforms=transform_train_pseudo, label_state=1)
            labeled_set = StandardSegmentationDataset(root=base, data_set=data_set,
                                                      image_set=(str(split) + '_labeled_' + str(sets_id)),
                                                      transforms=transform_train, label_state=0)
            pseudo_labeled_loader = torch.utils.data.DataLoader(dataset=pseudo_labeled_set,
                                                                batch_size=batch_size_pseudo,
                                                                num_workers=workers, shuffle=True)
            labeled_loader = torch.utils.data.DataLoader(dataset=labeled_set,
                                                         batch_size=batch_size_labeled,
                                                         num_workers=workers, shuffle=True)
            return labeled_loader, pseudo_labeled_loader, val_loader

        else:
            # Labeling
            unlabeled_set = StandardSegmentationDataset(root=base, data_set=data_set, mask_type='.npy',
                                                        image_set=(str(split_u) + '_unlabeled_' + str(sets_id)),
                                                        transforms=transform_test, label_state=2)
            unlabeled_loader = torch.utils.data.DataLoader(dataset=unlabeled_set, batch_size=batch_size_labeled,
                                                           num_workers=workers,
                                                           shuffle=False)
            return unlabeled_loader

        # Copied and modified from torch/vision/references/segmentation
def test_one_set(loader, device, net, categories, num_classes, output_size):
    # Evaluate on 1 data_loader (cudnn impact < 0.003%)
    net.eval()
    conf_mat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for image, target in tqdm(loader):
            image, target = image.to(device), target.to(device)
            result_val = net(image)
            output = result_val['out']

            output = torch.nn.functional.interpolate(output, size=output_size, mode='bilinear', align_corners=True)
            conf_mat.update(target.flatten(), output.argmax(1).flatten())

    acc_global, acc, iu = conf_mat.compute()
    print(categories)
    print((
        'global correct: {:.2f}\n'
        'average row correct: {}\n'
        'IoU: {}\n'
        'mean IoU: {:.2f}').format(
        acc_global.item() * 100,
        ['{:.2f}'.format(i) for i in (acc * 100).tolist()],
        ['{:.2f}'.format(i) for i in (iu * 100).tolist()],
        iu.mean().item() * 100))

    return acc_global.item() * 100, iu.mean().item() * 100, conf_mat.mat, iu