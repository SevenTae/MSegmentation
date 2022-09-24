import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn
from train_utils.trainModule.unet.util.dice_coefficient_loss import dice_loss, build_target
import  numpy as  np

'''训练过程的验证以及测试'''



def computDiceloss(inputs, target, num_classes, ignore_index):
    diceloss = 0
    dice_target = build_target(target, num_classes, ignore_index)
    diceloss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    return diceloss


def evaluateloss(net, dataloader, device,numclass=20, ignoreindex=100,isresize=None,Diceloss =False):

    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0


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
    valloss = val_loss / num_val_batches
    if Diceloss:
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
        return  valloss,diceloss
    else:
        return valloss




import train_utils.trainModule.unet.util.utils as  utils


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
        # confmat.compute()

    if isDice:
        header1 = 'Test Dice:'
        metric_logger2 = utils.MetricLogger(delimiter="  ")
        dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
        with torch.no_grad():
            for batch in metric_logger2.log_every(data_loader, 100, header1):
                image, target = batch['image'], batch['label']
                image = image.to(device=device, dtype=torch.float32)
                target = target.to(device=device, dtype=torch.long)
                model = model.to(device)
                output = model(image)
                dice.update(output, target)

        return confmat.re_zhib() ,dice.value.item() #返回confmat.re_zhib()acc_global, acc, iu,miou,
    return confmat.re_zhib()


def calculate_frequency_labels( dataloader, num_classes,ignor_index=255):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()

        mask = (y>=0) & (y< num_classes)&(y!=ignor_index)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes) #计算这个batch的中各个类别的像素数量
        # batch_total_frequency= np.sum(count_l)
        # batch_class_frequency=count_l/batch_total_frequency
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = frequency / total_frequency
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset+'_classes_weights.npy')
    # np.save(classes_weights_path, ret)

    return ret



def evalue_Fwiou(model, data_loader, device, num_classes,isResize=None,ignore_index =255):
    #我的计算方法：先算出整个测试集的类别概率，在计算整个测试集的iou 然后计算fwiou
    # 计算Fwiou
    print("计算每一类出现的频率")
    class_frequency = calculate_frequency_labels(data_loader,num_classes,ignore_index)
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header0 = 'Test FWiou:'
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
        acc_global, acc, iu, precion, recall, f1 = confmat.compute()
        fwiou =[]
        for  i in range(len(iu)):
            wiou= iu[i]*class_frequency[i]
            fwiou.append(wiou)
        fwiou=sum(fwiou)/len(fwiou)*100
        print("FWiou结束")
        return fwiou
