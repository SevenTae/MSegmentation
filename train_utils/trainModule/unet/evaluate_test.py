
import  argparse
import torch.nn.functional as F
import  numpy as np
import  random
import torch
from  .evaluate_train import evaluateloss,evalue_iou_miou_Dice,computDiceloss
import  time

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



