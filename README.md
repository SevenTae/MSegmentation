## 参考

https://github.com/bubbliiiing/unet-pytorch
https://github.com/milesial/Pytorch-UNet.git
https://github.com/jfzhang95/pytorch-deeplab-xception.git

因为刚用github不知道license怎么用，这里标出了参考，若有冒犯,请多海涵



trainModule里边存的是训练的模板，以此为模板区customer建立对应的文件见去改<br>
utils 文件夹里的utils_iou_miou 同上<br>
evaluate_module 存放的测评指标（iou, miou ..）的方法 来源于霹雳巴拉大佬和buling  他俩算出来的一样 一般用霹雳巴拉的就行<br>


模型训练好后再对应的文件夹里测评miou 等指标再根目录下的predict 可视化结果 以及测试速度<br>

<br>
集成的功能:<br>
   1.常用公开数据集和自定义数据集的载入（公开数据目前已完成cityscapes和pascal voc）<br>
   2.各种数据增强操作<br>
   3.不同模型的训练分离,具体表现为一个模型建立一个文件夹用于存放训练，验证，测试文件以及产生的训练日志和权重文件，
   其中训练日志支持wandb和tensorboard<br>
   4.小功能:模型参数和计算复杂度的计算，统计数据集的均值和方差，统计分割中各类别像素出现的频率以及形成类别权重
   5.分割的标签如果是单通道灰度图的化，支持单通道彩色化，onnx(感觉没用)<br>
   6.支持多种评价指标:iou ,miou,oa,准确率，召回率，dice系数，f1-score(后四个还没开始)<br>
   7.可视化,gradcam(有 但是不好用 正准备完善),hiddenlayer可视化网络结构（废了这个需要的依赖挺麻烦的）<br>
   8.支持半监督训练方式（准备拓展）