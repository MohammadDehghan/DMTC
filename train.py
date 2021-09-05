import time
import torch
from utils.common import ConfusionMatrix, save_checkpoint
from utils.module_list import *
from torch.cuda.amp import autocast, GradScaler


def train(exp_name, writer, loader_c, loader_sup, validation_loader, device, criterion, net, optimizer, lr_scheduler,
          num_epochs, is_mixed_precision, with_sup, num_classes, categories, input_sizes,
          sup_contrative, accum_iter, strong_threshold, temp, num_queries, num_negatives, val_num_steps=1000,
          loss_freq=10, tensorboard_prefix='', best_mIoU=0):
    #######
    # c for carry (pseudo labeled), sup for support (labeled with ground truth) -_-
    # Don't ask me why
    #######
    # Poly training schedule
    # Epoch length measured by "carry" (c) loader
    # Batch ratio is determined by loaders' own batch size
    # Validate and find the best snapshot per val_num_steps
    loss_num_steps = int(len(loader_c) / loss_freq)
    net.train()
    epoch = 0
    if with_sup:
        iter_sup = iter(loader_sup)

    if is_mixed_precision:
        scaler = GradScaler()

    accum_iter = accum_iter
    # Training
    running_stats = {'disagree': -1, 'current_win': -1, 'avg_weights': 1.0, 'loss': 0.0}
    while epoch < num_epochs:
        conf_mat = ConfusionMatrix(num_classes)
        time_now = time.time()
        for i, data in enumerate(loader_c, 0):
            # Combine loaders (maybe just alternate training will work)
            if with_sup:
                inputs_c, labels_c = data
                inputs_sup, labels_sup = next(iter_sup, (0, 0))
                if type(inputs_sup) == type(labels_sup) == int:
                    iter_sup = iter(loader_sup)
                    inputs_sup, labels_sup = next(iter_sup, (0, 0))

                # Formatting (prob: label + max confidence, label: just label)
                float_labels_sup = labels_sup.clone().float().unsqueeze(1)
                probs_sup = torch.cat([float_labels_sup, torch.ones_like(float_labels_sup)], dim=1)
                probs_c = labels_c.clone()
                labels_c = labels_c[:, 0, :, :].long()

                # Concatenating
                inputs = torch.cat([inputs_c, inputs_sup])
                labels = torch.cat([labels_c, labels_sup])
                probs = torch.cat([probs_c, probs_sup])

                probs = probs.to(device)
            else:
                inputs, labels = data

            # Normal training
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if is_mixed_precision:
            
                with autocast():
                    result = net(inputs)
                    outputs = result['out']
                    if sup_contrative:
                        rep = result['rep']
                        rep = torch.nn.functional.interpolate(rep, size=input_sizes[0], mode='bilinear', align_corners=True)
                    outputs = torch.nn.functional.interpolate(outputs, size=input_sizes[0], mode='bilinear', align_corners=True)
                    conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())

                    if with_sup:
                        loss, stats = criterion(outputs, probs, inputs_c.shape[0])

                        if sup_contrative:
                            with torch.no_grad():
                               
                                mask = F.interpolate((labels.unsqueeze(1) >= 0).float(), size=outputs.shape[2:], mode='nearest')
                                label = F.interpolate(label_onehot(labels, num_classes), size=outputs.shape[2:], mode='nearest')
                                prob = torch.softmax(outputs, dim=1)

                            sup_contrastive_loss = contrastive_loss(rep, label, mask, prob, strong_threshold, temp, num_queries, num_negatives)
                            loss = loss + 0.1 * sup_contrastive_loss
                    else:
                        loss, stats = criterion(outputs, labels)
                        if sup_contrative:
                            with torch.no_grad():
                                
                                mask = F.interpolate((labels.unsqueeze(1) >= 0).float(), size=outputs.shape[2:], mode='nearest')
                                label = F.interpolate(label_onehot(labels, num_classes), size=outputs.shape[2:], mode='nearest')
                                prob = torch.softmax(outputs, dim=1)

                            sup_contrastive_loss = contrastive_loss(rep, label, mask, prob, strong_threshold, temp, num_queries, num_negatives)
                            loss = loss + sup_contrastive_loss
            else:
                result = net(inputs)
                outputs = result['out']
                if sup_contrative:
                        rep = result['rep']
                        rep = torch.nn.functional.interpolate(rep, size=input_sizes[0], mode='bilinear', align_corners=True)
                outputs = torch.nn.functional.interpolate(outputs, size=input_sizes[0], mode='bilinear', align_corners=True)
                conf_mat.update(labels.flatten(), outputs.argmax(1).flatten())

                if with_sup:
                    loss, stats = criterion(outputs, probs, inputs_c.shape[0])
                    loss, stats = criterion(outputs, labels)
                    if sup_contrative:
                        with torch.no_grad():
                           
                            mask = F.interpolate((labels.unsqueeze(1) >= 0).float(), size=outputs.shape[2:], mode='nearest')
                            label = F.interpolate(label_onehot(labels, num_classes), size=outputs.shape[2:], mode='nearest')
                            prob = torch.softmax(outputs, dim=1)

                        sup_contrastive_loss = contrastive_loss(rep, label, mask, prob, strong_threshold, temp, num_queries, num_negatives)
                        loss = loss + 0.1 * sup_contrastive_loss
                else:
                    loss, stats = criterion(outputs, labels)
                    if sup_contrative:
                        with torch.no_grad():
                            
                            mask = F.interpolate((labels.unsqueeze(1) >= 0).float(), size=outputs.shape[2:], mode='nearest')
                            label = F.interpolate(label_onehot(labels, num_classes), size=outputs.shape[2:], mode='nearest')
                            prob = torch.softmax(outputs, dim=1)

                        sup_contrastive_loss = contrastive_loss(rep, label, mask, prob, strong_threshold, temp, num_queries, num_negatives)
                        loss = loss + sup_contrastive_loss


            if is_mixed_precision:

                loss = loss / accum_iter

                scaler.scale(loss).backward()
                if ((i +1) % accum_iter == 0) or (i+1 == len(loader_c)):
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
            else:
                if ((i +1) % accum_iter == 0) or (i+1 == len(loader_c)):
                    loss = loss / accum_iter
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

            lr_scheduler.step()

            # Logging
            for key in stats.keys():
                running_stats[key] += stats[key]
            current_step_num = int(epoch * len(loader_c) + i + 1)
            if current_step_num % loss_num_steps == (loss_num_steps - 1):
                for key in running_stats.keys():
                    print('[%d, %d] ' % (epoch + 1, i + 1) + key + ' : %.4f' % (running_stats[key] / loss_num_steps))
                    writer.add_scalar(tensorboard_prefix + key,
                                      running_stats[key] / loss_num_steps,
                                      current_step_num)
                    running_stats[key] = 0.0

            # Validate and find the best snapshot
            if current_step_num % val_num_steps == (val_num_steps - 1) or \
                current_step_num == num_epochs * len(loader_c) - 1:
                
                test_pixel_accuracy, test_mIoU, confmatrix, iou_per_class = test_one_set(loader=validation_loader, device=device, net=net,
                                                              num_classes=num_classes, categories=categories,
                                                              output_size=input_sizes[2])

                writer.add_scalar(tensorboard_prefix + 'test pixel accuracy',
                                  test_pixel_accuracy,
                                  current_step_num)
                writer.add_scalar(tensorboard_prefix + 'test mIoU',
                                  test_mIoU,
                                  current_step_num)
                net.train()

                # Record best model(Straight to disk)
                if test_mIoU > best_mIoU:
                    best_mIoU = test_mIoU
                    np.save('confusion_matrix/{}.npy'.format(exp_name), confmatrix.cpu())
                    with open('logging/' + exp_name + '.txt', 'a') as f:
                        line = 'categories: {} \n IoU: {} \n mean IoU: {}\n \n'.format(categories,['{:.2f}'.format(i) for i in (iou_per_class * 100).tolist()], test_mIoU)
                        f.write(line)
                    save_checkpoint(net=net, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                    is_mixed_precision=is_mixed_precision)

        # Evaluate training accuracies(same metric as validation, but must be on-the-fly to save time)
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

        train_pixel_acc = acc_global.item() * 100
        train_mIoU = iu.mean().item() * 100
        writer.add_scalar(tensorboard_prefix + 'train pixel accuracy',
                          train_pixel_acc,
                          epoch + 1)
        writer.add_scalar(tensorboard_prefix + 'train mIoU',
                          train_mIoU,
                          epoch + 1)

        epoch += 1
        print('Epoch time: %.2fs' % (time.time() - time_now))

    return best_mIoU