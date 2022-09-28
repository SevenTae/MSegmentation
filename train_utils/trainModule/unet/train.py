#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 
@File    : train_module
@Author  : root
@create_time    : 2022/9/13 22:57
训练文件模板
"""

import argparse
import logging
from .util.utils import getLogger
from pathlib import Path
import torch
import torch.nn as nn
# import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from  .evaluate_train import evaluateloss,evalue_iou_miou_Dice,computDiceloss,evalue_Fwiou,calculate_frequency_labels
import  time
from  nets.unet.unet_model import UNet,weights_init
import numpy  as np
"由于服务器用不了wandb所以这里删除了用wandb记录的代码"
import  os
import  numpy as np
import  random
import torch
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子


# 加了tensorboard记录
dir_checkpoint = Path('./checkpoints/')
tensorboarddir= Path('./TensorboardLog/')
logsavedir = Path('./logs/')





def  criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    total_loss =0.0
    total_loss = nn.functional.cross_entropy(inputs, target, ignore_index=ignore_index, weight=loss_weight)
    if dice is True:
        total_loss+= computDiceloss(inputs, target, num_classes, ignore_index)

    return  total_loss

##cityscpes 我直接改成20类了19类原来的+1类背景（原来255位置的）
def train_net(net,
              device,
              resume=False,
              isPretrain=False,
              epochs: int = 5,
              train_batch_size: int = 2,
              val_batch_size: int = 2,
              learning_rate: float = 1e-3,
              save_checkpoint: bool = True,
              ignoreindex: int = 100,
              num_class = 21,
              useDice=False,
              fwiou =None

              ):
    # 1. Create dataset
    # myself数据加载
    from dataloaders.datasets import pascal_customer2
    parsertrain = argparse.ArgumentParser()
    argstrain = parsertrain.parse_args()
    argstrain.resize = (256, 256)  # 输入是w h(长和高)


    parserval = argparse.ArgumentParser()
    argsval = parserval.parse_args()
    argsval.resize = (256, 256)  # 输入是w h(长和高)
    imgshape_base=None

    train_d = pascal_customer2.Customer_VOCSegmentation(argstrain, split='train',isAug=False)
    val_d = pascal_customer2.Customer_VOCSegmentation(argsval, split='val',)

    print("计算fwiou的每一类出现的概率")
    fwiou_label_weight = calculate_frequency_labels(val_d, num_classes=num_class, ignor_index=ignoreindex)



    n_val = val_d.__len__()
    n_train = train_d.__len__()

    train_loader = DataLoader(train_d, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=False,drop_last=True)
    val_loader = DataLoader(val_d, shuffle=True, num_workers=4, pin_memory=False, batch_size=val_batch_size,drop_last=True)


    # 初始化tensorboard
    # Path(tensorboarddir).mkdir(parents=True, exist_ok=True)
    workTensorboard_dir = os.path.join(dir_checkpoint,
                                       time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(workTensorboard_dir):
        os.makedirs(workTensorboard_dir)

    workcheckpoint_dir = os.path.join(dir_checkpoint,
                                      time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(workcheckpoint_dir):
        os.makedirs(workcheckpoint_dir)

    writer = SummaryWriter(workTensorboard_dir)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        train_Batch size:      {train_batch_size}
        val_Batch size:      {val_batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}


    ''')
    net = net.to(device)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9,weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98, )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2,eta_min=1e-6)
    # we = np.array([0.69, 1.34, 2.5, 2.31, 0.5, 0.84, 2.17, 0.87, 0.83, 1.45, 0.85, 2.38, 1.73], np.float32)
    # we = torch.from_numpy(we).to(device)


    start_epoch = 1
    #是否有预训练
    if isPretrain:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(isPretrain, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)
        print("加载成功")
        logging.info(f'Model loaded from {isPretrain}')
    else:#没有预训练使用pyroch的一些权重初始化方法
        strweight='normal'
        weights_init(net,init_type=strweight)
        logging.info(f'没有预训练权重，{strweight}权重初始化完成')
    #是否使用断点训练
    if resume:
        path_checkpoint = resume  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        # 加载优化器参数
        start_epoch = checkpoint['epoch'] +1 # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载lr_scheduler
    else:
        start_epoch= 1


    # 5. Begin training
    epochs_score = []  # 记录每个epoh的miou
    best_miou = 0.0  # 记录最好的那个

    time1 = time.time()
    start = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time1))
    print("训练{}个epoch的开始时间为:{}".format(epochs,start))
    logger = getLogger(logsavedir)
    logger.info("训练{}个epoch的开始时间为:{}".format(epochs,start))

    for epoch in range(start_epoch, epochs + 1):
        logger.info("train|epoch:{epoch}\t".format(epoch=epoch))
        current_miou = 0.0
        total_train_loss = 0
        net.train()
        print('Start Train')
        time1_1 = time.time()

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='batch') as pbar:
            for iteration, batch in enumerate(train_loader):  # batchsize=2 一共1487.5个batch
                images = batch['image']
                true_masks = batch['label']
                # true_masks =true_masks-1#这一条是针对有的8位彩色图索引从1开始
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                out = net(images)
                loss = 0.0
                if useDice:
                    loss = criterion(out, true_masks, loss_weight=None,  num_classes=num_class, dice= True, ignore_index = ignoreindex)

                else:
                    loss = criterion(out, true_masks, loss_weight=None, num_classes=num_class, dice=False,
                              ignore_index=ignoreindex)

                '''1.loss 2.梯度清零，3.反向传播。backward 4optomizer更新.'''

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update()  # 用来记录一次更新多少的
                total_train_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

        print("Finish Train")
        time1_2 = time.time()
        one_epoch_time = round((time1_2-time1_1)/60,3) #按照分钟来算
        print("训练一个epoch大概:%s Seconds"%(one_epoch_time))
        print('start Validation')
        # Evaluation round  #每个epoch评估一次
        isevalue = True
        if isevalue == True:
            current_lr= optimizer.param_groups[0]['lr']

            if useDice:
                # acc_global, acc, iu, miou ,Dice= evalue_iou_miou_Dice(net, val_loader, device, num_class,isResize=imgshape_base,isDice=True )
                Yuan,Dice= evalue_iou_miou_Dice(net, val_loader, device, num_class,isResize=imgshape_base,isDice=True )
                fwiou= evalue_Fwiou(net, val_loader, device, num_class,weight= fwiou_label_weight,isResize=imgshape_base)

                acc_global, acc, iu, precion, recall, f1, miou = Yuan
                print("我看看")
                val_score = miou  #这个看情况 有时候用dice
                val_loss,diceloss = evaluateloss(net, val_loader, device, numclass=num_class, ignoreindex=ignoreindex,isresize=imgshape_base,Diceloss=True)
                # logging.info('Validation loss : {}'.format(val_loss))
                # logging.info('Validation acc_global score: {}'.format(acc_global))
                # logging.info('Validation acc score: {}'.format(acc))
                # logging.info('Validation iu score: {}'.format(iu))
                # logging.info('Validation miou score: {}'.format(val_score)),
                # logging.info('Validation Dice score: {}'.format(Dice))
                # logging.info('本次训练时长: {} Seconds'.format(one_epoch_time))

                logger.info('Validation loss : {}'.format(val_loss))
                logger.info('Validation Dice loss : {}'.format(diceloss))
                logger.info('Validation acc_global score: {}'.format(acc_global))
                logger.info('Validation acc score: {}'.format(acc))
                logger.info('Validation iu score: {}'.format(iu))

                logger.info('Validation precion score: {}'.format(precion))
                logger.info('Validation recall score: {}'.format(recall))
                logger.info('Validation f1 score: {}'.format(f1))

                logger.info('Validation miou score: {}'.format(val_score))
                logger.info('Validation fwiou score: {}'.format(fwiou))

                logger.info('Validation Dice score: {}'.format(Dice))
                logger.info('本次训练时长: {} Seconds'.format(one_epoch_time))

                epochs_score.append(val_score)

                # tensorboard 记录
                #注意目前暂且iu f1这种单别数组的形式还进不了tensorboard
                writer.add_scalar("train_total_loss", total_train_loss /( iteration + 1), epoch)
                writer.add_scalar("lr", current_lr, epoch)

                writer.add_scalar("valloss", val_loss, epoch)
                writer.add_scalar("valmiou", val_score, epoch)
                writer.add_scalar("valfmiou", fwiou, epoch)

                writer.add_scalar("valDice", Dice, epoch)
                writer.add_scalar('best_epoch_index', epochs_score.index(max(epochs_score)) + 1, epoch)
            else:
                acc_global, acc, iu, precion, recall, f1, miou = evalue_iou_miou_Dice(net, val_loader, device, num_class,
                                                                       isResize=imgshape_base)
                val_score = miou  #
                fwiou = evalue_Fwiou(net, val_loader, device, num_class, isResize=imgshape_base)

                val_loss = evaluateloss(net, val_loader, device, numclass=num_class, ignoreindex=ignoreindex,
                                                  isresize=imgshape_base)
                logger.info('Validation loss : {}'.format(val_loss))
                logger.info('Validation acc_global score: {}'.format(acc_global))
                logger.info('Validation acc score: {}'.format(acc))
                logger.info('Validation iu score: {}'.format(iu))
                logger.info('Validation fwiou score: {}'.format(fwiou))


                logger.info('Validation precion score: {}'.format(precion))
                logger.info('Validation recall score: {}'.format(recall))
                logger.info('Validation f1 score: {}'.format(f1))
                logger.info('Validation miou score: {}'.format(val_score))

                logger.info('本次训练时长: {} Seconds'.format(one_epoch_time))

                current_miou = val_score
                epochs_score.append(val_score)

                # tensorboard 记录
                writer.add_scalar("train_total_loss", total_train_loss / (iteration + 1), epoch)
                writer.add_scalar("lr", current_lr, epoch)

                writer.add_scalar("valloss", val_loss, epoch)
                writer.add_scalar("valmiou", val_score, epoch)
                writer.add_scalar("valfmiou", fwiou, epoch)

                writer.add_scalar('best_epoch_index', epochs_score.index(max(epochs_score)) + 1, epoch)

        print('Finish Validation')
        scheduler.step()  # 这个地方是按照迭代来调整学习率的

        # 保存最好的miou和最新的
        if save_checkpoint:

            # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'lr_schedule': scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(workcheckpoint_dir, 'last.pth'))  # 保存最新的
            # 保存最好的
            if current_miou >= best_miou:
                best_miou = current_miou
                torch.save(checkpoint, os.path.join(workcheckpoint_dir, 'best.pth'))

    time2 = time.time()
    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time2))
    print("训练{}个epoch的结束时间为:{}".format(epochs, end))
    logger.info("训练{}个epoch的结束时间为:{}".format(epochs, end))
    writer.close()
    logger.info("训练完成")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--train_batch-size', '-tb', dest='train_batch_size', metavar='TB', type=int, default=2, help='Train_Batch size')
    parser.add_argument('--val_batch-size', '-vb', dest='val_batch_size', metavar='VB', type=int, default=2, help='Val_Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')  # 有没有预训练。。
    parser.add_argument('--ignore-index', '-i', type=int, dest='ignore_index', default=255,
                        help='ignore index defult 100')  # 有没有预训练。。
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--resume', '-r', type=str, default=False, help='is use Resume')
    parser.add_argument('--useDice', '-d', type=str, default=True, help='is use Dice')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    net = UNet( n_channels=3,n_classes= args.classes)

    try:
        train_net(net=net,
                  resume=args.resume,
                  epochs=args.epochs,
                  isPretrain=args.load,
                  train_batch_size=args.train_batch_size,
                  val_batch_size=args.val_batch_size,
                  learning_rate=args.lr,
                  device=device,
                  ignoreindex=args.ignore_index,
                  num_class = args.classes,
                  useDice=args.useDice,

                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
