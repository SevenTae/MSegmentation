#!/usr/bin/env python 
# -*- coding: utf-8 -*-
"""
@project: 评价指标管理
@File    : index_manager
@Author  : qiqq
@create_time    : 2022/11/2 18:44
"""

import logging
from pathlib import Path
import  os
import time


# def addLogger():
#
#
#
#     pass

class addLogger():

    def __init__(self,logger):
        self.logger =logger

    '''
     acc_global, class_acc, , precion, recall, f1,iu, miou,dice,fwiou,valloss
    '''

    def logger_index(self,index,indexstring=""):
        self.logger.info('Validation {} score: {}'.format(indexstring,index))
    #
    # def logger_acc_global(self,acc_global):
    #     self.logger.info('Validation acc_global score: {}'.format(acc_global))
    #
    # def logger_class_acc(self,class_acc):
    #     self.logger.info('Validation class_acc score: {}'.format(class_acc))
    #
    # def logger_precion(self,precion):
    #     self.logger.info('Validation precion score: {}'.format(precion))
    #
    # def logger_recall(self,recall):
    #     self.logger.info('Validation recall score: {}'.format(recall))
    #
    # def logger_f1(self,f1):
    #     self.logger.info('Validation f1 score: {}'.format(f1))
    #
    # def logger_iu(self,iu):
    #     self.logger.info('Validation iu score: {}'.format(iu))
    #
    # def logger_miou(self, miou):
    #     self.logger.info('Validation miou score: {}'.format(miou))
    #
    # def logger_dice(self, dice):
    #     self.logger.info('Validation dice score: {}'.format(dice))
    #
    # def logger_fwiou(self, fwiou):
    #     self.logger.info('Validation fwiou score: {}'.format(fwiou))
    #
    # def logger_valloss(self, valloss):
    #     self.logger.info('Validation valloss score: {}'.format(valloss))




class addTensorboard():

    def __init__(self, writer):
        self.writer = writer


    '''
     acc_global, class_acc, , precion, recall, f1,iu, miou,dice,fwiou,valloss
    '''

    def writer_singleindex(self, indexstring="", index=0,epoch=0):
        '''accglobal,fwiou,miou,,dice,valloss'''
        self.writer.add_scalar(indexstring, index, epoch)

    def writer_classindex(self, indexstring="",classindexs=[], epoch=0 ):
        '''classacc,precion,recall,,f1,iu'''
        for cla in range(len(classindexs)):
            self.writer.add_scalar('{}/class{}'.format(indexstring,cla), classindexs[cla], epoch)




