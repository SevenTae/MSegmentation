
import  argparse
import torch.nn.functional as F
import  numpy as np
import  random
import torch
from  .evaluate_train import evaluateloss,evalue_iou_miou_Dice,computDiceloss
import  time
from tqdm import  tqdm
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子

from .util import utils
from .util.dice_coefficient_loss import  dice_loss, build_target



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
            for image, target in metric_logger2.log_every(data_loader, 100, header1):
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
        _, _, iu=confmat.compute()
        fwiou =[]
        for  i in range(len(iu)):
            wiou= iu[i]*class_frequency[i]
            fwiou.append(wiou)
        fwiou = sum(fwiou) / len(fwiou) * 100
        print("FWiou结束")
        return fwiou





if __name__ == '__main__':
    #准备一个模型 准备一个dataloade

    from nets.unet.unet_model import UNet
    from dataloaders.datasets import  pascal_customer2
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1.准备数据
    parser2 = argparse.ArgumentParser()
    args2 = parser2.parse_args()


    args2.resize = (512, 512)  #
    imgbase =None
    batch_size = 2 #
    test = pascal_customer2.Customer_VOCSegmentation(args2, split='test')
    test_loader = DataLoader(test, shuffle=True, pin_memory=True, batch_size=batch_size)
    n_test = test.__len__()
    # 2.搭建网络

    model_path = r""
    model_dict = torch.load(model_path)
    model = UNet(3,21)
    model.load_state_dict(model_dict['net'])
    print("权重加载")

    useDice =False

    if useDice:
        ev,dice =evalue_iou_miou_Dice(model,test_loader,device,num_classes=13,isResize=imgbase,isDice=useDice)
        print("测试结果：")
        print(ev)
        print("dice:",dice)
    else:
        ev = evalue_iou_miou_Dice(model, test_loader, device, num_classes=13, isResize=imgbase)

    # filename = "tree数据集各种结果"
    #
    # with open(r"F:\MSegmentation\Test_log2\{}.txt".format(filename), mode='a', encoding='utf-8') as f:
    #     f.write("-------------------测试结果-------------------\n")
    #     f.write("{}\n".format(ev))
    # f.close()
    # print("结果已经保存")



