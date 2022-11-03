
from train_utils.trainModule.unet.util.dice_coefficient_loss import dice_loss, build_target
import train_utils.trainModule.unet.util.utils as  utils
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn


import  numpy as  np
'''训练过程的验证'''


def computDiceloss(inputs, target, num_classes, ignore_index):
    diceloss = 0
    dice_target = build_target(target, num_classes, ignore_index)
    diceloss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
    return diceloss


class Evalue():
    '''fwiou目前还不能用，没有想好一个简便的写法'''
    def __init__(self,model, data_loader, device, num_classes,weight,isResize=None,ignore_index =255):
       self.model=model
       self.data_loader=data_loader
       self.device=device
       self.num_classes=num_classes
       self.weight= weight
       self.isResize = isResize,
       if isinstance(self.isResize, tuple):
           self.isResize =  self.isResize[0]
       self.ignore_index = ignore_index


    def evalue_Valloss(self):

        #使用dice loss的话是dice+ce
        total_loss=0.0
        self.model.eval()
        num_val_batches = len( self.data_loader)
        val_loss = 0

        # iterate over the validation set
        for batch in tqdm( self.data_loader, total=num_val_batches, desc='Validation loss  round', unit='batch',
                          leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

            image, mask_true = batch['image'], batch['label']

            # move images and labelss to correct device and type
            image = image.to(device=self.device, dtype=torch.float32)
            mask_true = mask_true.to(device=self.device, dtype=torch.long)

            with torch.no_grad():
                # predict the mask
                mask_pred = self.model(image)

                # convert to one-hot format
                if self.num_classes == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                    pass
                else:  # 计算验证集的损失
                    #
                    criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

                    if self.isResize is not None:
                        mask_pred = F.interpolate(mask_pred, size=self.isResize, mode="bilinear", align_corners=True)
                        loss = criterion(mask_pred, mask_true)
                    else:
                        loss = criterion(mask_pred, mask_true)  # 计算完成一个batch的了
            val_loss += loss.item()

        # Fixes a potential division by zero error
        if num_val_batches == 0:
            return val_loss

        valloss = val_loss / num_val_batches
        total_loss= valloss
        return total_loss

    def evalue_CEDiceloss(self, Diceloss=True):

        # 使用dice loss的话是dice+ce
        total_loss = 0.0
        self.model.eval()
        num_val_batches = len(self.data_loader)
        val_loss = 0

        # iterate over the validation set
        for batch in tqdm(self.data_loader, total=num_val_batches, desc='Validation loss  round', unit='batch',
                          leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

            image, mask_true = batch['image'], batch['label']

            # move images and labelss to correct device and type
            image = image.to(device=self.device, dtype=torch.float32)
            mask_true = mask_true.to(device=self.device, dtype=torch.long)

            with torch.no_grad():
                # predict the mask
                mask_pred = self.model(image)

                # convert to one-hot format
                if self.num_classes == 1:  # 类别是1的时候 ，因为我们跑的是city所以不用管
                    pass
                else:  # 计算验证集的损失
                    #
                    criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

                    if self.isResize is not None:

                        mask_pred = F.interpolate(mask_pred, size=self.isResize, mode="bilinear", align_corners=True)
                        loss = criterion(mask_pred, mask_true)
                    else:

                        loss = criterion(mask_pred, mask_true)  # 计算完成一个batch的了
                val_loss += loss.item()
        valloss = val_loss / num_val_batches
        if Diceloss:
            dice_loss = 0
            for batch in tqdm(self.data_loader, total=num_val_batches, desc='Validation diceloss  round', unit='batch',
                              leave=False):  # 迭代完所有的验证集，输出整个验证集的loss

                image, target = batch['image'], batch['label']
                image = image.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)
                with torch.no_grad():
                    net = self.model.to(self.device)
                    output = net(image)
                    dloss = computDiceloss(output, target, self.num_classes, ignore_index=self.ignore_index)
                    dice_loss += dloss

            diceloss =dice_loss / num_val_batches
            total_loss = valloss + diceloss
            return total_loss
        else:
            total_loss = valloss
            return total_loss

    def evalue_IouMiouP(self):
        self.model.eval()
        confmat = utils.ConfusionMatrix(self.num_classes)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header0 = 'Test iou miou:'
        with torch.no_grad():
            for batch in metric_logger.log_every(self.data_loader, 100, header0):
                image, target = batch['image'], batch['label']
                image = image.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)
                # ！！注意有的数据集标签可能从1开开始 要把它弄成从0
                model = self.model.to(self.device)
                output = model(image)
                if self.isResize  is not None:
                    output = F.interpolate(output, size=self.isResize, mode="bilinear", align_corners=True)

                    output = torch.softmax(output, dim=1)
                    output = output.argmax(dim=1)
                else:
                    output = torch.softmax(output, dim=1)  # b，dm h w
                    output = output.argmax(1)
                confmat.update(target.flatten(), output.flatten())
            # confmat.compute()
        return confmat.re_zhib()


    def evalue_Dice(self):
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'TestDice:'
        dice = utils.DiceCoefficient(num_classes=self.num_classes, ignore_index=self.ignore_index)
        with torch.no_grad():
            for batch in metric_logger.log_every(self.data_loader, 100, header):
                image, target = batch['image'], batch['label']
                image = image.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)

                model =  self.model.to(self.device)
                output = model(image)
                dice.update(output, target)

        return  dice.value.item()  # )


    def calculate_frequency_labels( self):
        # Create an instance from the data loader
        z = np.zeros((self.num_classes,))
        # Initialize tqdm
        tqdm_batch = tqdm(self.dataloader)
        print('Calculating classes weights')
        for sample in tqdm_batch:
            y = sample['label']
            y = y.detach().cpu().numpy()

            mask = (y>=0) & (y< self.num_classes)&(y!=self.ignor_index)
            labels = y[mask].astype(np.uint8)
            count_l = np.bincount(labels, minlength=self.num_classes) #计算这个batch的中各个类别的像素数量
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



    def evalue_Fwiou(self):
        #我的计算方法：先算出整个测试集的类别概率，在计算整个测试集的iou 然后计算fwiou
        # 计算Fwiou

        class_frequency = self.weight
        self.model.eval()
        confmat = utils.ConfusionMatrix(self.num_classes)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header0 = 'Test FWiou:'
        with torch.no_grad():
            for batch in metric_logger.log_every(self.data_loader, 100, header0):
                image, target = batch['image'], batch['label']
                image = image.to(device=self.device, dtype=torch.float32)
                target = target.to(device=self.device, dtype=torch.long)
                # ！！注意有的数据集标签可能从1开开始 要把它弄成从0
                model = self.model.to(self.device)
                output = model(image)
                if self.isResize is not None:
                    output = F.interpolate(output, size=self.isResize, mode="bilinear", align_corners=True)

                    output = torch.softmax(output, dim=1)
                    output = output.argmax(dim=1)
                else:
                    output = torch.softmax(output, dim=1)  # b，dm h w
                    output = output.argmax(1)
                confmat.update(target.flatten(), output.flatten())
            acc_global, acc, iu, precion, recall, f1=confmat.compute()
            fwiou =[]
            for  i in range(len(iu)):
                wiou= iu[i]*class_frequency[i]
                fwiou.append(wiou)

            fwiou = sum(fwiou) * 100
            print("FWiou结束")
            return fwiou
