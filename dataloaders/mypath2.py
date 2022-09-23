class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'D:/MSegmentation/data/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return 'F:/openmmmlab/mmsegmentation/data/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif  dataset=='pascal_customer':
             return  '/home/qiqq/q3dl/datasets/pvTest/VOCdevkit/VOC2012/'
        elif dataset=='Customer':
            return 'D:/MSegmentation/data/Customer/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
# "/home/qiqq/q3dl/datasets/pvTest/VOCdevkit/"
# D:/Yan/MyExperiment/new/MSegmentation/datasets/