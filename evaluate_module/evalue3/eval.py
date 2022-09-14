import torch
import  argparse

import evaluate_module.evalue3.utils   as utils
def evaluate(model, data_loader, device, num_classes):
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
            target= target-1
            output = model(image)

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.compute()
    # return   confmat.re_zhib()

    return confmat


if __name__ == '__main__':
    #准备一个模型 准备一个dataloade

    from nets.unet.unet_model import UNet

    from netCompaire.unet import UNet

    from dataloaders.datasets import  pascal_customer
    from torch.utils.data import DataLoader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1.准备数据
    parser2 = argparse.ArgumentParser()
    args2 = parser2.parse_args()
    args2.base_size = 256  # 这玩意干啥的
    args2.crop_size = 256
    args2.resize = (256, 256)  # 把它缩放成原图训练时候的大小

    batch_size = 4  # 测试的时候batch直接等于图片综述
    test = pascal_customer.Customer_VOCSegmentation(args2, split='test')
    test_loader = DataLoader(test, shuffle=False, pin_memory=True, batch_size=batch_size)
    n_test = test.__len__()
    # 2.搭建网络


    model_path = r"/Customer/trainunet/checkpoints/best.pth"
    model_dict = torch.load(model_path)
    model = UNet(n_channels=3, n_classes=6)
    model.load_state_dict(model_dict['net'])

    model_path = r"/Customer/trainunet/checkpoints0/best.pth"
    model_dict = torch.load(model_path)
    model = UNet(n_channels=3, n_classes=6)
    model.load_state_dict(model_dict['nets'])

    print("权重加载")

    ev =evaluate(model,test_loader,device,num_classes=6)

    print(ev)

    print(ev)

    filename = "原unet的测评指标"

    with open(r"F:\MSegmentation\Test_log\{}.txt".format(filename), mode='w', encoding='utf-8') as f:
        f.write("-------------------测试结果-------------------\n")
        f.write("{}\n".format(ev))
    f.close()
    print("结果已经保存")

