#!/usr/bin/python3
GPU_MEM_MIN = 10000 #MB
GPU_MEM_USED_MAX = 100 #MB

import sys
import argparse
import os
import os.path
import datetime
from functools import reduce
import operator

from utils.utils_checkGPU import wait4FreeGPU
from utils.utils_data import SlicesOfSubject
from utils.utils_csv import parse_csv
from utils.utils_view import view_tensor_data, view_tensor_image
from utils.utils_locker import Locker
from trainer import Trainer, Logger, models

import torch
import torch.utils.data
import SimpleITK as sitk
import numpy as np

filelock_name='/tmp/waitGPU666.lock'

parser = argparse.ArgumentParser(description='PyTorch Cartilage Segmentation')

# logdir 
parser.add_argument('--logdir', metavar='DIR', type=str,
        default='/tmp/tensorboradtmplog',
        help='directory to save tensorboard logging file')

# job: classification or segmentation
parser.add_argument('--job', metavar='seg/cla', type=str,
        default='seg',
        help='what is your job? Classification(cla) or Segmentation(seg)?')

# Mode
## With checkpoint
parser.add_argument('--load', metavar='PATH',
        type=str, default=None, 
        help='path to a checkPoint which training resumed from or evalution')


# data augmentation

parser.add_argument('--crop', metavar='pixels',
        type=int, default=None,
        help='patch size')

parser.add_argument('--rotate', metavar='rad',
        type=float, default=5*3.14159265359/180,
        help='random rotate')

parser.add_argument('--spacing', metavar='mm',
        type=float, default=None,
        help='resize image to specific voxel spacing')

parser.add_argument('--border_ratio', metavar='ratio (fraction)',
        type=float, default=5.0/180,
        help='set image border ratio, border will be cropped randomly')

## save checkpoint after trainer
parser.add_argument('--save', metavar='/path/to/save/save',
        type=str, default=[], action='store',
        help='prefix where you want to save the checkpoint after training')

## New trainer
parser.add_argument('--arch', '-a', metavar='ARCH', default=None,
                    choices=list(models.keys()),
                    help='model architecture: ' +
                        ' | '.join(list(models.keys())))
parser.add_argument('--classes', metavar='0 C1 ... CN',
        type=int, default=None, nargs='*',
        help='classes to clasify')
parser.add_argument('--unet_channels', metavar='C_input C_layer1 ... C_layerN',
        type=int, default=None, nargs='*',
        help='channels of each unet layer')
parser.add_argument('-b', '--batch_size', default=None, type=int,
                    metavar='N', help='mini-batch size (default: 2)')

parser.add_argument('--optim', '--optimizer', default='SGD', type=str, \
        metavar='SGD/Adam', help='Optimizer, only SGD and Adam available now')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')

# to train
parser.add_argument('--training', metavar='CSV_PATH',
        type=str, default=None, action='append',
        help='path to a csv file indicating training set')
parser.add_argument('--epochs', default=100, type=int, metavar='NN',
        help='number of total epochs to run')
parser.add_argument('--interval', default=10, type=int, metavar='N',
        help='save interval (uint: epoch)')

# validate training progress
parser.add_argument('--validation',  metavar='CSV_PATH',
        type=str, default=None, action='append',
        help='path to a csv file indicating validation set')

# to generate test results
parser.add_argument('--test',  metavar='CSV_PATH',
        type=str, default=None, action='append',
        help='path to a csv file indicating test set')

# check data?
parser.add_argument('--reviewData', metavar='True|False',
        type=bool, default=False,
        help='manually check all data')

# auxiliary
# worker threads
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')


def main():
    args = parser.parse_args()
    print('=> arguments:')
    print(args)

    # logger
    logger = Logger(args.logdir)
    # prepare model
    assert args.batch_size > 0
    assert args.workers >= 0

    locker = Locker(filelock_name)
    if not locker.lock():
        print('=> waiting behind the queue')
        locker.lock_block()
    print('=> locked, checking & waiting for available devices')
    GPU = wait4FreeGPU(GPU_MEM_MIN, GPU_MEM_USED_MAX, \
            wall=-1, interval=30, double_check=2)
    print('done, will use GPU ' + str(GPU))
    os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU)

    if args.load:
        print("=> loading checkpoint from {}".format(args.load))
        if os.path.isfile(args.load):
            trainer = Trainer(args.load)
        else:
            raise FileNotFoundError(\
                    'no checkpoint found at {}'.format(args.load))
    else:
        print("=> creating new model")
        assert len(args.classes) > 1
        assert args.lr != None
        model_name = args.arch
        model_param = {'classes_num': len(args.classes)}
        if model_name == 'unets':
            model_param['channels']=args.unet_channels
        optimizer_name = args.optim
        optimizer_param = \
                {'lr': args.lr,\
                'weight_decay': args.weight_decay}
        if optimizer_name == 'SGD':
            optimizer_param['momentum'] = 0.9
        criterion_name = "CrossEntropyLoss"
        accuracy_name = "Top1"
        accuracy_param = {}
        classes = args.classes

        trainer = Trainer(
            model_name = model_name,
            model_param = model_param,
            optimizer_name = optimizer_name,
            optimizer_param = optimizer_param,
            criterion_name = criterion_name,
            accuracy_name = accuracy_name,
            accuracy_param = accuracy_param,
            classes = classes)
    print('=> trainer prepared') 
    print(trainer)

    locker.unlock()
    print('=> locker released') 

    # prepare data
    if args.training:
        print("=> loading training set {}".format(args.training), \
                "batch_size={}, workers={}, rotate={}, crop={}, \
spacing={}, border_ratio={}".format(args.batch_size, args.workers, \
                args.rotate, args.crop, args.spacing, args.border_ratio))
        training_rows = reduce(operator.add, map(parse_csv, args.training))
        training_list = list(map(lambda row, args=args: SlicesOfSubject( \
                sitk.ReadImage(row[1]), row[2],\
                classes = trainer.classes, job = args.job, \
                spacing=np.array((args.spacing, args.spacing)), \
                crop=np.array((args.crop, args.crop)), \
                ratio=np.array((args.border_ratio, args.border_ratio)), \
                rotate=args.rotate, \
                include_slices=row[3]), training_rows))
        training_dataset = \
                torch.utils.data.dataset.ConcatDataset(training_list)
        # sampling weights
        #print("=> constructing weighted sampler")
        #labels = [label for (_, label) in  training_dataset]
        #hist, _ = np.histogram(labels, \
        #        bins=len(args.classes), range=(-0.5, len(args.classes)-0.5))
        #print("=> distribution: {}".format(str(hist)))
        #weights = map(\
        #        lambda x, prob=np.reciprocal(hist,dtype=np.float64): prob[x], \
        #        labels)
        #training_sampler = torch.utils.data.sampler.WeightedRandomSampler( \
        #        list(weights), len(labels))
        #training_loader = torch.utils.data.DataLoader( \
        #        training_dataset, sampler=training_sampler, \
        #        batch_size=args.batch_size, shuffle=False, \
        #        num_workers=args.workers, pin_memory=True)
        training_loader = torch.utils.data.DataLoader( \
                training_dataset, \
                batch_size=args.batch_size, shuffle=True, \
                num_workers=args.workers, pin_memory=True)
        # review data
        if args.reviewData:
            for i, onedataset in enumerate(training_list):
                for j in range(len(onedataset)):
                    data = onedataset[j]
                    view_tensor_image(data[0], 'training_'+str(i)+'_'+str(j))

    if args.validation:
        print("=> loading validation set {}\nworkers={}"\
                .format(args.validation, args.workers))
        validation_rows=reduce(operator.add, map(parse_csv, args.validation))
        validation_list = list(map(lambda row: SlicesOfSubject( \
                sitk.ReadImage(row[1]), row[2],\
                classes = trainer.classes, job = args.job, \
                spacing=np.array((args.spacing, args.spacing)), \
                crop=np.array((args.crop, args.crop)), \
                include_slices=row[3]), validation_rows))
        validation_dataset = \
                torch.utils.data.dataset.ConcatDataset(validation_list)
        validation_loader = torch.utils.data.DataLoader(
                validation_dataset,
                batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        # review data
        if args.reviewData:
            for i, onedataset in enumerate(validation_list):
                for j in range(len(onedataset)):
                    data = onedataset[j]
                    view_tensor_image(data[0], \
                            'validation_'+str(i)+'_'+str(j))


    #cudnn.benchmark = True

    # train
    if args.training is not None:
        print("=> start training for {} epochs, saved each {} epochs".format(\
                args.epochs, args.interval))
        for cnt in range(args.epochs):
            trainer.train(training_loader, logger)
            if args.validation:
                trainer.validate(validation_loader, logger)
            if (cnt+1)%args.interval == 0:
                checkpoint = args.save+str(trainer.epoch)+'.pkl'
                print("=> saving checkpoint to {}".format(checkpoint))
                trainer.save(checkpoint)
    
    # validate only
    if (args.validation is not None) and (args.training is None):
        print("=> start validation")
        trainer.validate(validation_loader, logger)

    if args.test:
        print("=> start test with {}".format(args.test))
        test_rows = reduce(operator.add, map(parse_csv, args.test))
        for row in test_rows:
            print("=> evaling subject {}".format(row[1]))
            test_dataset = SlicesOfSubject( \
                    sitk.ReadImage(row[1]), None,\
                    classes = trainer.classes, job = args.job, \
                    spacing=np.array((args.spacing, args.spacing)), \
                    crop=np.array((args.crop, args.crop)), \
                    include_slices=row[3])
            test_loader = torch.utils.data.DataLoader( \
                    test_dataset, \
                    batch_size = args.batch_size, shuffle=False, \
                    num_workers=args.workers, pin_memory=False, \
                    drop_last=False)
            # review data
            if args.reviewData:
                for i, onedataset in enumerate(test_list):
                    for j in range(len(onedataset)):
                        data = onedataset[j]
                        view_tensor_data(data, trainer.classes, \
                                'test_'+str(i)+'_'+str(j))
            if args.job == 'seg':
                result = trainer.eval(test_loader, logger)
                print('=> & saving result to {}'.format(row[2]))
                test_dataset.setPrediction(\
                        list(range(len(test_dataset))), result)
                sitk.WriteImage(test_dataset.getPrediction(), row[2])
            if args.job == 'cla':
                result = trainer.eval(test_loader, logger, raw=True)
                indice = list(map(lambda x: (x+1), \
                        test_dataset.slice_indice))
                _, top1 = torch.max(result, dim=1)
                result = result.numpy()
                #top1 = top1.numpy().tolist()
                #test_dataset.setPrediction(\
                #        list(range(len(test_dataset))), top1)
                #prediction = (test_dataset.getPrediction()[ \
                #        test_dataset.slice_indice]).tolist()
                for i in range(len(top1)):
                    print('=> result:' + row[0] + ',' + row[1] + \
                        ',' + str(indice[i]) + ',' + str(row[2]) + \
                        ',' + ' '.join(map(str,result[i,:].tolist())))
    print("=> all done, exit now!")


if __name__ == '__main__':
    main()
