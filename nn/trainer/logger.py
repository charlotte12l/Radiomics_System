#!/usr/bin/python3
import datetime
import numpy as np
from . import tb_logger 

class Meter(object):
    """Computes and stores the avg/max/min and current value"""
    def __init__(self):
        super(Meter, self).__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = 0
        self.min = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.max = val
            self.min = val
        else:
            self.max = max(self.max, val)
            self.min = min(self.min, val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, loggerdir):
        super(Logger, self).__init__()
        self.logger = tb_logger.Logger(loggerdir)
        self.time = Meter()
        self.time_data = Meter()

    def iter_hook_train(self, **state):
        state['mode'] = 'train'
        self.iter_hook(**state)

    def iter_hook_valid(self, **state):
        state['mode'] = 'valid'
        self.iter_hook(**state)

    def iter_hook_eval(self, **state):
        state['mode'] = 'eval'

        mode = state['mode']
        epoch = state['epoch']
        i = state['dataset_loader_index']
        l = state['dataset_loader_length']
        batch_size = state['batch_size']
        if i == 0:
            self.time.reset()
            self.time_data.reset()
        self.time.update(state['time'])
        self.time_data.update(state['time_data'])

        report = "{:<6}".format(mode)
        report += "[{0}]{3}[{1}/{2}] ".format(epoch, i+1, l, batch_size)
        # last iteration of a epoch
        if (i+1) == l:
            report += "ET:{} ".format(
                    datetime.timedelta(seconds=round(self.time.sum)))
            report = "{:<80}".format(report)
            print(report, flush=True)
        else:
            report += "data/time(avg):{:.2f}({:.1f})/{:.2f}({:.1f}) ".format(
                    self.time_data.val, self.time_data.avg,
                    self.time.val, self.time.avg)
            report += "ETR:{} ".format(
                    datetime.timedelta(\
                            seconds=round(self.time.sum/(i+1)*(l-i-1))))
            report = "{:<80}".format(report)
            print(report, end='\r', flush=True)
        return

    def iter_hook(self, **state):
        '''a callback function which will be executed
            at the end of each mini_batch training/validation.
        expected keys:
            mode(train, valid, test),
            dataset_loader_index, dataset_loader_length, batch_size,
            loss, prec, time, time_data
        You may define your own logger to implement more functions,
            such as make a class to draw or some other things
        '''
        mode = state['mode']
        epoch = state['epoch']
        i = state['dataset_loader_index']
        l = state['dataset_loader_length']
        batch_size = state['batch_size']
        loss = state['loss'].numpy()
        assert loss.ndim == 0
        prec = state['prec'].numpy()
        assert prec.ndim <= 2
        if i == 0:
            self.time.reset()
            self.time_data.reset()
            self.precA=prec.copy()
        else:
            self.precA = np.concatenate((self.precA, prec.copy()))
        self.time.update(state['time'])
        self.time_data.update(state['time_data'])

        self.logger.scalar_summary(tag=mode+'/Loss', value=loss.mean(), \
                step=(epoch-1)*l+i)

        if prec.ndim == 1:
            self.logger.scalar_summary(tag=mode+'/Acc', value=prec.mean(), \
                step=(epoch-1)*l+i)
        else:
            # prec.ndim == 2
            for index in range(0, prec.shape[1]):
                self.logger.scalar_summary(tag=mode+'/Acc/'+str(index), \
                        value=prec[:,index].mean(), step=(epoch-1)*l+i)

        report = "{:<6}".format(mode)
        report += "[{0}]{3}[{1}/{2}] ".format(
                epoch, i+1, l, batch_size)
        # last iteration of a epoch
        if (i+1) == l:
            report += "ET:{} ".format(
                    datetime.timedelta(seconds=round(self.time.sum)))
            # display stage as tensorboard did
            stage = (epoch-1)*l+i
            if stage < 1e3:
                report += "ST:{} ".format(stage)
            elif stage < 1e6:
                report += "ST:{:.4g}k".format(stage/1e3)
            #else:
            #    report += "ST:{0:.3e} ".format(stage)
            report = "{:<80}".format(report)
            print(report, flush=True)
            if self.precA.ndim == 1:
                self.logger.histo_summary( \
                        tag=mode+'/histAcc', values=self.precA, step=epoch, \
                        bins=256)
            else:
                for index in range(0, self.precA.shape[1]):
                    self.logger.histo_summary( \
                            tag=mode+'/histAcc/'+str(index), \
                            values=self.precA[:,index], step=epoch, \
                            bins=256)
        else:
            report += "data/time(avg):{:.2f}({:.1f})/{:.2f}({:.1f}) ".format(
                    self.time_data.val, self.time_data.avg,
                    self.time.val, self.time.avg)
            report += "ETR:{} ".format(
                    datetime.timedelta(\
                            seconds=round(self.time.sum/(i+1)*(l-i-1))))
            report = "{:<80}".format(report)
            print(report, end='\r', flush=True)
        return

