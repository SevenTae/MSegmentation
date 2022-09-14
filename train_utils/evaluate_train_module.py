import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from .dice_coefficient_loss import dice_loss, build_target

'''训练过程的验证以及测试'''


#训练过程中验证集的loss
def computDiceloss(inputs, target, num_classes, ignore_index):
    diceloss = 0
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss= 0
        dice_target = build_target(target, num_classes, ignore_index)
        diceloss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
    return dice_loss



def evaluateloss(net, dataloader, device,numclass=20, ignoreindex=100,isresize=None,isonlyDice =False):

    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0

    if isonlyDice:
        diceloss = 0
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation diceloss  round', unit='batch',
                          leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

            image, target = batch['image'], batch['label']
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.long)
            with torch.no_grad():
                net = net.to(device)
                output = net(image)
                loss = computDiceloss(output, target,numclass, ignore_index=ignoreindex)
            diceloss +=loss
        return  diceloss

    else:
        # iterate over the validation set
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation loss  round', unit='batch',
                          leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

            image, mask_true = batch['image'], batch['label']

            # move images and labelss to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            with torch.no_grad():
                # predict the mask
                mask_pred = net(image)

                # convert to one-hot format
                if numclass == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                    pass
                else:  # 计算验证集的损失
                    #
                    criterion = nn.CrossEntropyLoss(ignore_index=ignoreindex)

                    if isresize is not None:

                        mask_pred = F.interpolate(mask_pred, size=isresize, mode="bilinear", align_corners=True)
                        loss = criterion(mask_pred, mask_true)
                    else:

                        loss = criterion(mask_pred, mask_true)  # 计算完成一个batch的了
            val_loss += loss.item()

        # Fixes a potential division by zero error
        if num_val_batches == 0:
            return val_loss
        # print("valloss(total val):", val_loss / num_val_batches)
        return val_loss / num_val_batches  # 然后再区batch的平均




import train_utils.utils as  utils

#测试的时候的时候使用
def evalue_iou_miou_Dice(model, data_loader, device, num_classes,isResize=None,isDice =False,ignore_index =255):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header0 = 'Test iou miou:'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header0):
            image, target = batch['image'], batch['label']
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.long)
            # ！！注意有的数据集标签可能从1开开始 要把它弄成从0
            model = model.to(device)
            output = model(image)
            if isResize is not None:
                output = F.interpolate(output, size=isResize, mode="bilinear", align_corners=True)

                output = torch.softmax(output, dim=1)
                output = output.argmax(dim=1)
            else:
                output = torch.softmax(output, dim=1)  # b，dm h w
                output = output.argmax(1)
            confmat.update(target.flatten(), output.flatten())
        confmat.compute()

    if isDice:
        diceloss=0
        header1 = 'Test Dice:'
        metric_logger2 = utils.MetricLogger(delimiter="  ")
        with torch.no_grad():
            for image, target in metric_logger2.log_every(data_loader, 100, header1):
                image, target = batch['image'], batch['label']
                image = image.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.long)
                model = model.to(device)
                output = model(image)
                for name, x in output.items():
                    dice_target = build_target(target, num_classes, ignore_index)
                diceloss +=dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)

        return confmat.re_zhib() ,diceloss #返回acc_global, acc, iu,miou
    return confmat.re_zhib()