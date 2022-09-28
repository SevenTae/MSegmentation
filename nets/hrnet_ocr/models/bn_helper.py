import torch
import functools


#有病吧能照顾照顾我们这些只有一张卡的吗
# startswith() 方法用于检查字符串是否是以指定子字符串开头
if torch.__version__.startswith('0'):
    pass
    #不用管 这个是关于低版本的 多卡并行 不用管
    # from .sync_bn.inplace_abn.bn import InPlaceABNSync
    # BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    # BatchNorm2d_class = InPlaceABNSync
    # relu_inplace = False
else:
    # BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    BatchNorm2d_class = BatchNorm2d = torch.nn.BatchNorm2d
    relu_inplace = True