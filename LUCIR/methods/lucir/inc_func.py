import torch
import numpy as np
import torch.nn as nn
from utils.misc import *
import wandb
from torch.nn import functional as F
# from arptrainer.pearson_loss import pearson_loss
import matplotlib.pyplot as plt
from utils.misc import AverageMeter


def make_batch_one_hot(input_tensor, n_classes: int, dtype=torch.float):
    targets = torch.zeros(input_tensor.shape[0], n_classes, dtype=dtype)
    targets[range(len(input_tensor)), input_tensor.long()] = 1
    return targets

def incremental_train_and_eval(args, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, trainloader, testloader,
            iteration, start_iteration, the_lambda, fix_bn=True, weight_per_class=None, device=None):
    # Setting up the CUDA device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        # ref_model.eval()
        ref_model.train()

        num_old_classes = ref_model.fc.out_features

    for epoch in range(args.num_epochs):

        # Start training for the current phase, set the two branch models to the training mode
        tg_model.train()

        # Fix the batch norm parameters according to the config
        if fix_bn:
            for m in tg_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # Set all the losses to zeros
        train_loss = 0
        train_loss_ce, train_loss_3, train_loss_kd = AverageMeter(), AverageMeter(), AverageMeter()

        # Set the counters to zeros
        correct = 0
        total = 0

        # Learning rate decay
        tg_lr_scheduler.step()
        # Print the information
        print('\nEpoch: %d, learning rate: ' % epoch, end='')
        print(tg_lr_scheduler.get_lr()[0])
        # wandb.log({
        #     'epoch':epoch,
        #     'lr':tg_lr_scheduler.get_lr()[0]
        # })

        for batch_idx, (inputs, targets, path) in enumerate(trainloader):
            # Get a batch of training samples, transfer them to the device
            inputs, targets = inputs.to(device), targets.to(device)
            if iteration == start_iteration:
                # import pdb;pdb.set_trace()
                outputs = tg_model(inputs)
                features = outputs['features']
                logits = outputs['logits']

                loss_ce = nn.CrossEntropyLoss(weight_per_class)(logits, targets)
                train_loss_ce.update(loss_ce.item(), targets.size(0))
                loss = loss_ce
            else:
                outputs = tg_model(inputs)
                cur_features = outputs['features']
                logits = outputs['logits']
                with torch.no_grad():
                    ref_outputs = ref_model(inputs)
                    ref_features = ref_outputs['features']

                output_binary = torch.zeros(logits.size(0), 2).to(device)
                task_num = int(logits.size(1) / 2)
                if args.binary_loss == 'sum_b_sig': ##sum before sigmoid
                    for i in range(task_num):
                        output_binary[:, 0] += nn.Sigmoid()(logits[:, i * 2])
                        output_binary[:, 1] += nn.Sigmoid()(logits[:, i * 2 + 1])
                    output_binary = torch.sigmoid(output_binary)
                    y_binary = make_batch_one_hot(targets%2, 2).to(device)
                    loss_binary = nn.BCELoss()(output_binary, y_binary)
                    # loss_binary = nn.CrossEntropyLoss(weight_per_class)(output_binary, targets%2)
                    loss_ce = (1 - args.binary_weight) * nn.CrossEntropyLoss(weight_per_class)(logits, targets) + \
                              args.binary_weight * loss_binary
                elif args.binary_loss == 'sum_a_sig': ##sum after sigmoid  # sum logits
                    for i in range(task_num):
                        output_binary[:, 0] += nn.Sigmoid()(logits[:, i * 2])
                        output_binary[:, 1] += nn.Sigmoid()(logits[:, i * 2 + 1])
                    output_binary = output_binary / torch.sum(output_binary, 1, keepdim=True)
                    # loss_binary = nn.CrossEntropyLoss(weight_per_class)(output_binary, targets%2)
                    y_binary = make_batch_one_hot(targets%2, 2).to(device)

                    loss_binary = nn.BCELoss()(output_binary, y_binary)
                    loss_ce = (1 - args.binary_weight) * nn.CrossEntropyLoss(weight_per_class)(logits, targets) + \
                              args.binary_weight * loss_binary
                elif args.binary_loss == 'sum_b_log': ##sum outside log #sum features
                    for i in range(task_num):
                        output_binary[:, 0] += torch.log(nn.Sigmoid()(logits[:, i * 2]))
                        output_binary[:, 1] += torch.log(nn.Sigmoid()(logits[:, i * 2 + 1]))
                    output_binary = output_binary / torch.sum(output_binary, 1, keepdim=True)
                    # loss_binary = ((1 - targets%2).mul(output_binary[:, 0]) + (targets%2).mul(output_binary[:, 1]))\
                    #               / (logits.size(0))
                    loss_binary = ((1 - targets%2).mul(output_binary[:, 0]) + (targets%2).mul(output_binary[:, 1])).sum() / (logits.size(0))
                    loss_ce = (1 - args.binary_weight) * nn.CrossEntropyLoss(weight_per_class)(logits, targets) + \
                              args.binary_weight * loss_binary
                elif args.binary_loss == 'none': ## no use
                    loss_ce = nn.CrossEntropyLoss(weight_per_class)(logits, targets)
                else: ##max
                    output_real = torch.zeros(logits.size(0), task_num)
                    output_fake = torch.zeros(logits.size(0), task_num)
                    for i in range(task_num):
                        output_real[:, i] = nn.Sigmoid()(logits[:, i * 2])
                        output_fake[:, i] = nn.Sigmoid()(logits[:, i * 2 + 1])
                    output_max_real, _ = torch.max(output_real, 1)
                    output_max_fake, _ = torch.max(output_fake, 1)
                    output_binary[:, 0] = output_max_real
                    output_binary[:, 1] = output_max_fake
                    output_binary = output_binary / torch.sum(output_binary, 1, keepdim=True)
                    y_binary = make_batch_one_hot(targets%2, 2).to(device)
                    loss_binary = nn.BCELoss()(output_binary, y_binary)
                    # loss_binary = nn.CrossEntropyLoss(weight_per_class)(output_binary, targets%2)
                    loss_ce = (1 - args.binary_weight) * nn.CrossEntropyLoss(weight_per_class)(logits, targets) + \
                           args.binary_weight * loss_binary

                train_loss_ce.update(loss_ce.item(), targets.size(0))

                # Loss 2: feature-level distillation loss
                loss_kd = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                                                 torch.ones(inputs.shape[0]).to(device)) * the_lambda
                train_loss_kd.update(loss_kd.item(), targets.size(0))
                #
                # # import pdb;pdb.set_trace()
                # # Loss 3: margin ranking loss
                outputs_bs = logits
                assert (outputs_bs.size() == logits.size())
                gt_index = torch.zeros(outputs_bs.size()).to(device)
                gt_index = gt_index.scatter(1, targets.view(-1, 1), 1).ge(0.5)
                gt_scores = outputs_bs.masked_select(gt_index)
                max_novel_scores = outputs_bs[:, num_old_classes:].topk(args.K, dim=1)[0]
                hard_index = targets.lt(num_old_classes)
                hard_num = torch.nonzero(hard_index).size(0)
                if hard_num > 0:
                    gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, args.K)
                    max_novel_scores = max_novel_scores[hard_index]
                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    loss3 = nn.MarginRankingLoss(margin=args.dist)(gt_scores.view(-1, 1), max_novel_scores.view(-1, 1),
                                                              torch.ones(hard_num * args.K).to(device)) * args.lw_mr
                else:
                    loss3 = torch.zeros(1).to(device)
                train_loss_3.update(loss3.item(), targets.size(0))
                loss = loss_kd + loss_ce + loss3


            # Backward and update the parameters
            tg_optimizer.zero_grad()
            loss.backward()
            tg_optimizer.step()

            # Record the losses and the number of samples to compute the accuracy
            train_loss += loss.item()
            _, predicted = (logits).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Print the training losses and accuracies
        print('Train set: {}, train loss: {:.4f}, train_loss_ce: {:.4f}, train_loss_kd: {:.4f}, train_loss_3: {:.4f}, '
              'accuracy: {:.4f}'.format(
            len(trainloader), train_loss / (batch_idx + 1), train_loss_ce.avg, train_loss_kd.avg, train_loss_3.avg,
            100. * correct / total))

        # wandb.log({
        #     'epoch': epoch,         'batch': batch_idx,
        #     'train loss ce':        train_loss_ce.avg,
        #     'train loss arp':       train_loss_arp.avg,
        #     'train loss upcenter':  train_loss_upcenter.avg,
        #     'train loss pearson':   train_loss_pearson.avg,
        #     'train loss r':         train_loss_r.avg,
        #     'train loss':           train_loss / (batch_idx + 1),
        #     'train accuracy':       100. * correct / total,
        #     'train accuracy arp':   100. * correct_arp / total,
        # })


        # Running the test for this epoch
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                logits = outputs['logits']
                loss = nn.CrossEntropyLoss(weight_per_class)(logits, targets)
                test_loss += loss
                _, predicted = (logits).max(1)
                total += targets.size(0)
                # correct += predicted.eq(targets).sum().item()
                correct += (predicted%2).eq((targets%2)).sum().item()

        print('Test set: {} test loss: {:.4f} fc accuracy: {:.4f}'.format(len(testloader),
                        test_loss / (batch_idx + 1), 100. * correct / total))
        # wandb.log({
        #     'epoch': epoch, 'batch': batch_idx,
        #     'test loss': test_loss / (batch_idx + 1),
        #     'test accuracy': 100. * correct / total,
        # })

    return tg_model
