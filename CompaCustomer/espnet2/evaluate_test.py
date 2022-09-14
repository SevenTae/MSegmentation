
import  argparse
import torch.nn.functional as F
import  numpy as np
import  random
import torch
seed = 7
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子


import  utils.utils_iou_miou as utils
def evaluate(model, data_loader, device, num_classes,isResize=None):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 100, header):
            image, target = batch['image'], batch['label']
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.long)
            #！！注意有的数据集标签可能从1开开始 要把它弄成从0
            model = model.to(device)
            # target= target-1
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
    # return   confmat.re_zhib()

    return confmat


if __name__ == '__main__':
    #准备一个模型 准备一个dataloade

    from netCompaire.ESPV2.SegmentationModel import EESPNet_Seg
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

    model_path = r"F:\MyReSegmet\compaCustome2\espnet2\checkpoints\best.pth"
    model_dict = torch.load(model_path)
    model = EESPNet_Seg(classes=13, s=2)
    model.load_state_dict(model_dict['net'])
    print("权重加载")

    ev =evaluate(model,test_loader,device,num_classes=13,isResize=imgbase)
    print("测试结果：")
    print(ev)

    # filename = "tree数据集各种结果"
    #
    # with open(r"F:\MSegmentation\Test_log2\{}.txt".format(filename), mode='a', encoding='utf-8') as f:
    #     f.write("-------------------测试结果-------------------\n")
    #     f.write("{}\n".format(ev))
    # f.close()
    # print("结果已经保存")



