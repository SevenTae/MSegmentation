#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 20:51
# @Author  : xx
import logging
import os
import time

def getLogger(savedir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)
    work_dir="./"

    # FileHandler
    work_dir = os.path.join(savedir,
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w',encoding="utf-8")
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger
import time
if __name__ == '__main__':
    # 获取logger
    logger = getLogger(".")
    time1 = time.time()
    start = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time1))
    logger.info("训练{}个epoch的开始时间为:{}".format(1000, start))
    for  i in  range(1000):
        logger.info('train|epoch:{epoch}\tstep:{step}/{all_step}\tloss:{loss:.4f}'.format(epoch=1, step=1 + 1,
                                                                                          all_step=5,
                                                                                          loss=5))  # 打印训练日志
    time2 = time.time()
    end = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time2))
    logger.info("训练{}个epoch的结束时间为:{}".format(1000, end))