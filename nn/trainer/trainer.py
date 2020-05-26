import time
import torch
import torch.nn
import torch.nn.functional

from .models import models
from .losses import losses
from .optimizers import optimizers


# try to add .cpu() and .gpu() method if possible
class Trainer(object):
    # self.training_log is a dict,
    #   keys are training epoch, va
    simple_var_names = [
            # params 
            'epoch', 'classes',
            # auxiliary
            # names
            'model_name', 'criterion_name', 'accuracy_name', 'optimizer_name',
            # init params
            'model_param', 'criterion_param',
            'accuracy_param','optimizer_param']
    
    module_names = ['model', 'optimizer', 'criterion', 'accuracy']

    def __init__(self, *args, **kw):
        '''Trainer initialization function.
        You can either
        1. load a saved trainer (checkpoint) or
        2. create a new trainer.
        To load a checkpoint, use same argument as help(Trainer.load).
            For example:
            trainer = Trainer('path_to_save_trainer.pkl')
        To create a new trainer, use same arguement as help(Trainer.new).
            For example:
            trainer = Trainer(
                    model_name='unet', model_param = {'classes_num': 2},
                    optimizer_name = 'SGD',
                    optimizer_param = {'lr': 0.1, 'momentum': 0.9},
                    criterion_name = 'CrossEntropyLoss',
                    accuracy_name = 'CrossEntropyLoss',
                    classes = [0, 3])
        '''
        if len(args) == 1 and len(kw) == 0:
            # load/resume trainer from file like object
            self.load(args[0])
        elif len(args) == 0 and len(kw) > 0:
            # create new trainer with arguments key pairs
            self.new(**kw)
        else:
            raise ValueError(\
                    'invalid argument, '+\
                    'try help(trainer.Trainer.__init__) for more info')
        #super(object, self).__init__()
        # Modules
        # for name in module_names:
        #    self.__dict__[name] = None
        # simple variables
        # for name in simple_var_names:
        #    self.__dict__[name] = None

    def is_inited(self):
        init_state = True
        for name in self.module_names + self.simple_var_names:
            init_state &= name in self.__dict__
        return init_state

    def __str__(self):
        report = 'trainer:' + "\n"
        if not self.is_inited():
            report += "not initialized" + "\n"
            report += "use new or load method to init trainer" + "\n"
            return report
        for name in self.simple_var_names:
            report += name + "\t: "
            report += str(self.__dict__[name])
            report += "\n"
        #for name in self.module_names:
        #    report += name + ' state' + "\t: "
        #    report += self.__dict__[name].state_dict()['param_groups']
        #    report += "\n"
        # report only optimizer in modules
        #report += 'optimizer' + "\t: "
        #report += str(self.optimizer.state_dict()['param_groups'])
        #report += "\n"
        return report

    def _create_modules(self, resume=False):
        # create modules
        model = models[self.model_name](**self.model_param)
        # parallelize model
        model = torch.nn.DataParallel(model)
        self.model = model.cuda()
        self.optimizer = optimizers[self.optimizer_name](\
                filter(lambda x: x.requires_grad, self.model.parameters()), \
                **self.optimizer_param)
        self.criterion = losses[self.criterion_name](\
                **self.criterion_param).cuda()
        self.accuracy = losses[self.accuracy_name](\
                **self.accuracy_param).cuda()

    def new(self, **args):
        '''Check Trainer.simple_var_names for args keys needed, except -->
        'epoch' is not compulsory will be set to 0 even if it is specified.
        *_param is not compulsory and will be set to {} if not specified.
        '''
        args['epoch'] = 0
        #self.training_log = []
        for name in self.module_names:
            if not ((name+'_param') in args):
                args[name+'_param'] = {}

        for name in self.simple_var_names:
            self.__dict__[name] = args[name]
        # create model, optimizer, criterion, accuracy
        self._create_modules(resume=False)

    def load(self, f):
        '''Load trainer from save trainer (checkpoint).
        f: a file-like object (has to implement fileno that returns a file descriptor, and must implement seek), or a string containing a file name
        '''
        checkpoint = torch.load(f)
        #self.training_log = checkpoint['training_log']
        for name in self.simple_var_names:
            self.__dict__[name] = checkpoint[name]
        # create modules
        self._create_modules(resume=True)
        for name in self.module_names:
            self.__dict__[name].load_state_dict(checkpoint[name+'_state'])

    def save(self, f):
        state = {}
        # add training log
        #state['training_log'] = self.training_log
        # add simple vars
        for name in self.simple_var_names:
            state[name] = self.__dict__[name]
        # module state
        for name in self.module_names:
            state[name+'_state'] = self.__dict__[name].state_dict()
        # save
        torch.save(state, f)

    def train(self, training_loader, logger = None):
        '''For more about logger, see trainer.Logger,
            method iter_hook_train will be called.
        '''
        self.epoch = self.epoch + 1

        epoch = self.epoch
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        accuracy = self.accuracy

        # switch to train mode
        model.train()

        #losses = torch.cuda.FloatTensor(len(training_loader))
        #precs = torch.cuda.FloatTensor(len(training_loader))

        end = time.time()
        for i, (input, target) in enumerate(training_loader):
            # measure data loading time
            data_time = time.time() - end
    
            # async=True saves time transfering data to gpu
            target = target.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy when logger exists
            if logger:
                with torch.no_grad():
                    prec = accuracy(output, target)

            ## measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            if logger:
                loss = loss.cpu().detach()
                prec = prec.cpu().detach()

                logger.iter_hook_train(
                        epoch=self.epoch,
                        dataset_loader_index=i,
                        dataset_loader_length=len(training_loader),
                        batch_size = training_loader.batch_size,
                        loss=loss,
                        prec=prec,
                        time=batch_time,
                        time_data=data_time)


    def validate(self, val_loader, logger = None):
        '''For more about logger, see trainer.Logger,
            method iter_hook_valid will be called.
        '''
        model = self.model
        criterion = self.criterion
        accuracy = self.accuracy

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            data_time = time.time() - end
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.no_grad():
                output = model(input)
                loss = criterion(output, target)
                # measure accuracy when logger exists
                if logger:
                    prec = accuracy(output, target)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if logger:
                loss = loss.cpu().detach()
                prec = prec.cpu().detach()

                logger.iter_hook_valid(
                        epoch=self.epoch,
                        dataset_loader_index=i,
                        dataset_loader_length=len(val_loader),
                        batch_size = val_loader.batch_size,
                        loss=loss,
                        prec=prec,
                        time=batch_time,
                        time_data=data_time)

    def eval(self, test_loader, logger = None, raw=False):
        '''For more about logger, see trainer.Logger,
            method iter_hook_valid will be called.
        '''
        model = self.model
        criterion = self.criterion

        # switch to evaluate mode
        model.eval()

        result = None
        end = time.time()
        for i, (input, _) in enumerate(test_loader):
            data_time = time.time() - end

            input = input.cuda(non_blocking=True)
            with torch.no_grad():
                output = model(input)
                if raw:
                    predicted = torch.nn.functional.softmax(output, dim=1)
                else:
                    _, predicted = torch.max(output, 1)
            predicted = predicted.cpu().detach()
            if result is None:
                result = predicted
            else:
                result = torch.cat((result, predicted), dim=0)
    
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if logger:
                logger.iter_hook_eval(
                        epoch=self.epoch,
                        dataset_loader_index=i,
                        dataset_loader_length=len(test_loader),
                        batch_size = test_loader.batch_size,
                        time=batch_time,
                        time_data=data_time)

        return result

