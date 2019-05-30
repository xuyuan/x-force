
import os
import shutil

from pathlib import Path
from datetime import datetime
import yaml
import argparse
import socket
import warnings

import torch
from torch.utils.data.dataloader import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as vutils

from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from .optim.lr_scheduler import FindLR, NoamLR, WarmUpLR
from .mixup import mixup
from .utils import choose_device, get_num_workers


def create_optimizer(net, name, learning_rate, weight_decay, momentum=0, apex_opt_level=None,
                     optimizer_state=None, device=None, no_bn_wd=False, local_rank=None, sync_bn=False):
    net.float()

    if local_rank is not None:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        print('torch.distributed.init_process_group')
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', local_rank)
    else:
        device = choose_device(device)

    if sync_bn:
        import apex
        print("using synced BN")
        net = apex.parallel.convert_syncbn_model(net)

    print('use', device)
    net = net.to(device)

    # optimizer
    parameters = [p for p in net.parameters() if p.requires_grad]
    if no_bn_wd:
        parameters = bnwd_optim_params(net, parameters)

    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adamw':
        from .optim.adam import Adam
        optimizer = Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'fused_adam':
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adabound':
        from trainer.optim.adabound import AdaBound
        optimizer = AdaBound(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adaboundw':
        from trainer.optim.adabound import AdaBoundW
        optimizer = AdaBoundW(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(name)

    if apex_opt_level:
        from apex import amp
        net, optimizer = amp.initialize(net, optimizer, opt_level=apex_opt_level)

        def backward(loss):
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        optimizer.backward = backward
    else:
        optimizer.backward = lambda loss: loss.backward()

    if optimizer_state:
        # if use_fp16 and 'optimizer_state_dict' not in optimizer_state:
        #     # resume FP16_Optimizer.optimizer only
        #     optimizer.optimizer.load_state_dict(optimizer_state)
        # elif not use_fp16 and 'optimizer_state_dict' in optimizer_state:
        #     # resume optimizer from FP16_Optimizer.optimizer
        #     optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
        # else:
        optimizer.load_state_dict(optimizer_state)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if local_rank is not None:
            if apex_opt_level:
                import apex
                net = apex.parallel.DistributedDataParallel(net, delay_allreduce=True)
            else:
                net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        else:
            net = torch.nn.DataParallel(net)

    return net, optimizer


def create_lr_scheduler(optimizer, lr_scheduler, **kwargs):
    lr_scheduler_name = lr_scheduler
    if kwargs.get('local_rank', None) is not None:
        assert lr_scheduler_name not in ('plateau', 'findlr')  # TODO

    if not isinstance(optimizer, torch.optim.Optimizer):
        # assume FP16_Optimizer
        optimizer = optimizer.optimizer

    warmup_steps = kwargs['lr_scheduler_warmup']
    eta_min = kwargs['stopping_learning_rate']

    if lr_scheduler_name == 'plateau':
        patience = kwargs.get('lr_scheduler_patience', 10) // kwargs.get('validation_interval', 1)
        factor = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, eps=0)
    elif lr_scheduler_name == 'step':
        step_size = kwargs['lr_scheduler_step_size']
        gamma = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_scheduler_name == 'cos':
        max_epochs = kwargs['max_epochs'] - warmup_steps
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epochs, eta_min=eta_min)
    elif lr_scheduler_name == 'milestones':
        milestones = kwargs['lr_scheduler_milestones']
        gamma = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif lr_scheduler_name == 'findlr':
        max_steps = kwargs['max_steps']
        lr_scheduler = FindLR(optimizer, max_steps)
    elif lr_scheduler_name == 'noam':
        lr_scheduler = NoamLR(optimizer, warmup_steps=warmup_steps)
    else:
        raise NotImplementedError("unknown lr_scheduler " + lr_scheduler_name)

    if warmup_steps > 0 and lr_scheduler_name != 'noam':
        lr_scheduler = WarmUpLR(lr_scheduler, warmup_steps, eta_min)

    lr_scheduler.name = lr_scheduler_name
    return lr_scheduler


# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
# code adopted from https://github.com/diux-dev/imagenet18/blob/master/training/experimental_utils.py
def bnwd_optim_params(model, model_params):
    bn_params, remaining_params = split_bn_params(model, model_params)
    return [{'params': bn_params, 'weight_decay': 0}, {'params': remaining_params}]


def split_bn_params(model, model_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum

    mod_bn_params = get_bn_params(model)

    bn_params = [p for p in model_params if p in mod_bn_params]
    rem_params = [p for p in model_params if p not in mod_bn_params]
    return bn_params, rem_params


def create_dataset_loaders(dataset, args, use_cuda, shuffle):
    num_workers = get_num_workers(args.jobs)

    if args.local_rank is not None:
        shuffle = False
        sampler = DistributedSampler(dataset)
        ngpus_per_node = torch.cuda.device_count()
        num_workers = num_workers // ngpus_per_node
        batch_size = args.batch_size // ngpus_per_node
    else:
        sampler = None
        batch_size = args.batch_size

    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else default_collate

    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, sampler=sampler,
                             num_workers=num_workers, drop_last=True,
                             collate_fn=collate_fn, pin_memory=use_cuda)

    return data_loader


def targets_to_cuda(targets, non_blocking=True):
    if isinstance(targets, torch.Tensor):
        targets = targets.cuda(non_blocking=non_blocking)
    elif isinstance(targets, list):
        targets = [targets_to_cuda(t, non_blocking=non_blocking) for t in targets]
    elif isinstance(targets, dict):
        targets = {k: targets_to_cuda(v, non_blocking=non_blocking) for k, v in targets.items()}
    else:
        raise NotImplementedError(type(targets))
    return targets


def optimizer_cpu_state_dict(optimizer):
    # save cuda RAM
    optimizer_state_dict = optimizer.state_dict()

    dict_value_to_cpu = lambda d: {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                   for k, v in d.items()}

    if 'optimizer_state_dict' in optimizer_state_dict:
        #  FP16_Optimizer
        cuda_state_dict = optimizer_state_dict['optimizer_state_dict']
    else:
        cuda_state_dict = optimizer_state_dict

    if 'state' in cuda_state_dict:
        cuda_state_dict['state'] = {k: dict_value_to_cpu(v)
                                    for k, v in cuda_state_dict['state'].items()}

    return optimizer_state_dict


class Trainer(object):
    def __init__(self, model, datasets, criterion, args):
        """
        :param model: `torch.nn.Module` to be trained
        :param datasets: dict of datasets including 'train', 'valid', and 'test'
        :param criterion: callable loss function, returns dict of losses
        :param args: parsed results of `ArgumentParser`
        """
        self.is_main_process = args.local_rank is None or args.local_rank == 0
        self.datasets = datasets
        log_dir = Path(args.log_dir)

        # iteration counters
        self.iteration = 0
        self.start_epoch = 0
        self.min_epoch_loss = float('inf')
        self.max_metric_score = 0
        optimizer_state = None
        lr_scheduler_state = None
        self.args = args

        if self.is_main_process:
            # print args
            args_yaml = yaml.dump((vars(args)))
            terminal_columns = shutil.get_terminal_size().columns
            self.println("=" * terminal_columns)
            self.println(args_yaml + ("=" * terminal_columns))

        if args.local_rank is not None:
            assert args.device in ('auto', 'cuda')
            torch.cuda.set_device(args.local_rank)
            args.device = 'cuda'  # only support GPU

        if args.resume:
            self.println('resume checkpoint ...')
            resume_checkpoint = torch.load(args.resume_checkpoint_file, map_location=lambda storage, loc: storage)
            model = model.load(resume_checkpoint['model_file'])
            self.start_epoch = resume_checkpoint['epoch']
            self.min_epoch_loss = resume_checkpoint.get('min_epoch_loss', self.min_epoch_loss)
            self.max_metric_score = resume_checkpoint.get('max_metric_score', self.max_metric_score)
            self.iteration = resume_checkpoint['iteration']
            optimizer_state = resume_checkpoint['optimizer']
            lr_scheduler_state = resume_checkpoint['lr_scheduler']
            self.println('resume epoch {} iteration {}'.format(self.start_epoch, self.iteration))

        device = choose_device(args.device)
        self.use_cuda = device.type == 'cuda'

        self.mixup_epochs = args.no_mixup_epochs if args.no_mixup_epochs > 1.0 else (1 - args.no_mixup_epochs) * args.max_epochs
        self.criterion = criterion
        self.model = model
        self.net, self.optimizer = create_optimizer(model,
                                                    args.optim, args.learning_rate, args.weight_decay, args.momentum,
                                                    args.apex_opt_level,
                                                    optimizer_state=optimizer_state,
                                                    device=device,
                                                    no_bn_wd=args.no_bn_wd,
                                                    local_rank=args.local_rank,
                                                    sync_bn=args.sync_bn)

        self.data_loaders = {k: create_dataset_loaders(d, args, self.use_cuda, shuffle=(k=='train'))
                             for k, d in datasets.items() if k != 'test'}

        self.lr_scheduler = create_lr_scheduler(self.optimizer, **vars(args))
        if lr_scheduler_state:
            print(f'resume lr_scheduler_state {lr_scheduler_state}')
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
            print(f'resumed lr_scheduler_state{self.lr_scheduler.state_dict()}')

        self.checkpoints_folder = log_dir / 'checkpoints'
        self.checkpoints_folder.mkdir(parents=True, exist_ok=True)

        datasets_text = '\n '.join(['{} {}'.format(k, v) for k, v in datasets.items()])
        self.println("datasets:\n")
        self.println(datasets_text)

        self.tb_writer = None
        if self.is_main_process:
            print("logging into {}".format(log_dir))
            self.tb_writer = SummaryWriter(log_dir=str(log_dir))
            self.tb_writer.add_text('args', repr(args_yaml)[1:-1], 0)
            self.tb_writer.add_text('datasets', repr(datasets_text)[1:-1], 0)
            with (log_dir / 'args.yml').open('w') as f:
                f.write(args_yaml)
            #tb_writer.add_text('cfg', str(ssd_net) + "\n" + str(ssd_net.cfg), 0)

            #if not use_fp16:
            if args.write_graph:
                # write graph
                with torch.no_grad():
                    images = next(iter(self.data_loaders['train']))['image']
                    images = images.to(device)
                    model.trace_mode = True
                    self.tb_writer.add_graph(model, images)
                    model.trace_mode = False

    def println(self, *args, **kwargs):
        if self.is_main_process:
            print(*args, **kwargs)

    def run_epoch(self, epoch, phase):
        data_loader = self.data_loaders[phase]
        if isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)
            # loss counters
        epoch_loss_dict = {}

        is_train = phase == 'train'
        if is_train:
            if self.lr_scheduler.name != 'plateau':
                self.lr_scheduler.step(epoch=epoch)
            self.optimizer.zero_grad()
            if self.tb_writer:
                self.tb_writer.add_scalar('learning_rate', self.get_lr(), epoch)

        self.net.train(is_train)
        torch.set_grad_enabled(is_train)

        desc = f"Epoch {epoch} {phase}"
        if self.args.local_rank is not None:
            desc = f"[{self.args.local_rank}]" + desc
        pbar_disable = False if epoch == self.start_epoch + 1 else None
        pbar = tqdm(data_loader, desc=desc, unit="images",
                    unit_scale=data_loader.batch_size, leave=False, disable=pbar_disable,
                    mininterval=10, smoothing=1)
        it = 0
        # for logging images
        min_loss_in_epoch = float("inf")
        max_loss_in_epoch = 0
        batch_of_min_loss_in_epoch = None
        batch_of_max_loss_in_epoch = None

        for batch in pbar:
            inputs = batch.pop('input', None)
            targets = batch
            if inputs is None:
                warnings.warn(f'no input, skip (data in batch are {batch.keys()})')
                assert False
                continue

            if self.use_cuda:
                inputs = inputs.cuda(non_blocking=True)
                targets = targets_to_cuda(targets)

            criterion = self.criterion
            if phase == 'train' and self.args.mixup > 0:
                if epoch < self.mixup_epochs:
                    inputs, criterion = mixup(inputs, alpha=self.args.mixup, criterion=criterion)

            # forward
            outputs = self.net(inputs)
            losses = criterion(outputs, targets)

            # compute overall loss if multi losses is returned
            if isinstance(losses, dict):
                if 'All' not in losses:
                    losses['All'] = sum(losses.values())
            elif isinstance(losses, torch.Tensor):
                losses = dict(All=losses)
            else:
                raise RuntimeError(type(losses))
            loss = losses['All']
            optimize_step = False
            if phase == 'train':
                self.optimizer.backward(loss / self.args.gradient_accumulation)
                if self.iteration % self.args.gradient_accumulation == 0:
                    optimize_step = True
                    it += self.args.gradient_accumulation
                    if self.args.clip_grad_norm > 0:
                        clip_grad_norm_(self.net.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.iteration += 1
                if self.lr_scheduler and self.lr_scheduler.name == 'findlr':
                    self.lr_scheduler.step(self.iteration)
                    if self.tb_writer:
                        self.tb_writer.add_scalar('learning_rate', self.get_lr(), self.iteration)
            elif phase == 'valid':
                it += 1

            if self.args.local_rank is not None:
                # sync loss between processes
                world_size = torch.distributed.get_world_size()
                for l in losses.values():
                    torch.distributed.reduce(l, dst=0)
                    if self.is_main_process:
                        l /= world_size

            if not self.is_main_process:
                continue
            # Below are logging in optimization step

            batch_loss_dict = {k: v.item() for k, v in losses.items()}

            if self.tb_writer and optimize_step and self.args.log_loss_interval > 0 and self.iteration % self.args.log_loss_interval == 0:
                # tb_writer.add_scalars('Loss', batch_loss_dict, iteration)
                for k, v in batch_loss_dict.items():
                    self.tb_writer.add_scalar(phase + '/Loss/' + k, v, epoch)

            epoch_loss_dict = {k: epoch_loss_dict.get(k, 0) + v for k, v in batch_loss_dict.items()}

            batch_loss = batch_loss_dict['All']
            if batch_loss < min_loss_in_epoch:
                min_loss_in_epoch = batch_loss
                batch_of_min_loss_in_epoch = (inputs, targets)
            if batch_loss > max_loss_in_epoch:
                max_loss_in_epoch = batch_loss
                batch_of_max_loss_in_epoch = (inputs, targets)

            if it > 0:
                # update the progress bar
                scalars = {k: "%.03f" % (v/it) for k, v in epoch_loss_dict.items()}
                pbar.set_postfix(scalars, refresh=False)

        if not self.is_main_process:
            return 0

        epoch_loss_dict = {k: v/it for k, v in epoch_loss_dict.items()}
        if self.tb_writer:
            if self.args.log_images:
                name_batch = {"min_loss": batch_of_min_loss_in_epoch, "max_loss": batch_of_max_loss_in_epoch}
                for name, batch in name_batch.items():
                    if batch is not None:
                        images = self.visualize_batch(*batch)
                        images_grid = vutils.make_grid(images, normalize=False)
                        self.tb_writer.add_image('/'.join([phase, name]), images_grid, epoch)


            #scalars = {phase + k: v for k, v in epoch_loss_dict.items()}
            #tb_writer.add_scalars('EpochLoss', scalars, epoch)
            for k, v in epoch_loss_dict.items():
                self.tb_writer.add_scalar(phase + '/EpochLoss/' + k, v, epoch)

        return epoch_loss_dict['All']

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, epoch, model_filename, checkpoint_filename=None):
        if not checkpoint_filename:
            checkpoint_filename = model_filename
        model_filename = str(self.checkpoints_folder / model_filename) + '.model.pth'
        checkpoint_filename = str(self.checkpoints_folder / checkpoint_filename) + '.checkpoint.pth'

        self.model.save(model_filename)

        optimizer_state_dict = optimizer_cpu_state_dict(self.optimizer)

        torch.save({'epoch': epoch,
                    'min_epoch_loss': self.min_epoch_loss,
                    'max_metric_score': self.max_metric_score,
                    'iteration': self.iteration,
                    'optimizer': optimizer_state_dict,
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'model_file': model_filename,
                    'args': self.args
                    },
                   checkpoint_filename)

        checkpoint_saved = Path(checkpoint_filename)
        last_checkpoint_file = self.checkpoints_folder / 'last.checkpoint'
        if last_checkpoint_file.exists():
            last_checkpoint_file.unlink()
        last_checkpoint_file.symlink_to(checkpoint_saved.relative_to(self.checkpoints_folder))

    def run(self):
        self.println('Training', repr(self.model), 'Epochs:', self.start_epoch, '/', self.args.max_epochs)
        pbar_epoch = trange(self.start_epoch + 1, self.args.max_epochs + 1,
                            unit="epoch", disable=not self.is_main_process)

        for epoch in pbar_epoch:
            epoch_state = {}
            for phase in self.data_loaders:
                if phase == 'valid' and epoch % self.args.validation_interval != 0:
                    continue

                epoch_loss = self.run_epoch(epoch, phase)

                evaluation = None
                if 'test' in self.datasets and phase == 'valid':
                    evaluation = self.test()

                if not self.is_main_process:
                    continue
                # Below are processing between epoch, e.g. save checkpoints, logging, etc.

                early_stopping = False
                if evaluation is not None:
                    epoch_state['metric'] = metric_score = evaluation['score']
                    for k, v in evaluation.items():
                        if isinstance(v, dict) and 'score' in v:
                            self.tb_writer.add_scalar('test/' + k.replace(' ', '_'), v['score'], epoch)

                    if metric_score > self.max_metric_score:
                        self.max_metric_score = metric_score
                        print('\nsave checkpoint at epoch {} with best {} metric {}'.format(epoch, phase, self.max_metric_score))
                        self.save_checkpoint(epoch, "best_metric")

                if phase == 'valid' or 'valid' not in self.data_loaders:
                    if self.min_epoch_loss > epoch_loss:
                        self.min_epoch_loss = epoch_loss
                        print('\nsave checkpoint at epoch {} with best {} loss {}'.format(epoch, phase, self.min_epoch_loss))
                        self.save_checkpoint(epoch, 'best_loss')

                    if epoch % self.args.validation_interval == 0:
                        if self.args.lr_scheduler == 'plateau':
                            self.lr_scheduler.step(metrics=epoch_loss)

                        early_stopping = (self.get_lr() < self.args.stopping_learning_rate)

                if (early_stopping or (epoch == self.args.max_epochs) or (phase == 'valid') or
                        (self.args.checkpoints_interval > 0 and epoch % self.args.checkpoints_interval == 0 and
                         epoch % self.args.validation_interval != 0 )):
                    print('\nsave checkpoint at epoch {} with {} loss {}'.format(epoch, phase, epoch_loss))
                    self.save_checkpoint(epoch, "last")

                epoch_state[phase + '_loss'] = epoch_loss

                if early_stopping:
                    print('early stopping!')
                    print('Metric Score = {}'.format(self.max_metric_score))
                    return

                if self.args.lr_scheduler == 'findlr':
                    print('finish find lr')
                    return

            if self.is_main_process:
                epoch_state['time'] = datetime.now().strftime('%d%b%H:%M')
                epoch_state['min_loss'] = self.min_epoch_loss
                epoch_state['lr'] = self.get_lr()
                pbar_epoch.set_postfix(epoch_state, refresh=False)

        self.println('Metric Score = {}'.format(self.max_metric_score))

    def test(self):
        raise NotImplementedError()

    def visualize_batch(self, inputs, targets):
        raise NotImplementedError()


class ArgumentParser(object):
    def __init__(self, description=__doc__):
        self.parser = argparse.ArgumentParser(description=description,
                                              formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                              fromfile_prefix_chars='@')
        group = self.parser.add_argument_group('general options')
        group.add_argument('--validation-interval', type=int, default=5, help='interval of epochs for validation')
        group.add_argument("--mixup", type=float, default=-1, help='mixup alpha hyperparameter, negative for disabling')
        group.add_argument('--no-mixup-epochs', type=float, default=0.1,
                           help='Disable mixup training if enabled in the last N (or %%) epochs.')
        group.add_argument('--apex-opt-level', default=None, choices=('O0', 'O1', 'O2', 'O3'),
                           help='different pure and mixed precision modes from apex')
        group.add_argument('--sync-bn', action='store_true', help='enabling sync BatchNorm')

        group = self.parser.add_argument_group('options of devices')
        group.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')
        group.add_argument("--local_rank", type=int, default=None, help='args for torch.distributed.launch module')

        group = self.parser.add_argument_group('options of dataloader')
        group.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
        group.add_argument('--jobs', default=-2, type=int,
                           help='How many subprocesses to use for data loading. ' +
                                'Negative or 0 means number of cpu cores left')

        group = self.parser.add_argument_group('options of optimizer')
        group.add_argument("--optim", default='sgd',
                           choices=['sgd', 'adam', 'adamw', 'fused_adam', 'adabound', 'adaboundw'],
                           help='choices of optimization algorithms')
        group.add_argument('--max-epochs', default=100, type=int, help='Number of training epochs')
        group.add_argument('--learning-rate', default=1e-3, type=float, help='initial learning rate')
        group.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD optimizer')
        group.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay')
        group.add_argument('--no-bn-wd', action='store_true', help='Remove batch norm from weight decay')
        group.add_argument('--gradient-accumulation', type=int, default=1,
                           help='accumulate gradients over number of batches')
        group.add_argument('--clip-grad-norm', type=float, default=0,
                           help='clips gradient norm of model parameters.')

        group = self.parser.add_argument_group('options of learning rate scheduler')
        group.add_argument("--lr-scheduler", default='milestones', help='method to adjust learning rate',
                           choices=['plateau', 'step', 'milestones', 'cos', 'findlr', 'noam', 'cosw'])
        group.add_argument("--lr-scheduler-patience", type=int, default=20,
                           help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
        group.add_argument("--lr-scheduler-step-size", type=int, default=40,
                           help='lr scheduler step: number of epochs of learning rate decay.')
        group.add_argument("--lr-scheduler-gamma", type=float, default=0.1,
                           help='learning rate is multiplied by the gamma to decrease it')
        group.add_argument("--lr-scheduler-milestones", type=lambda s: [int(item) for item in s.split(',')],
                           default=[60, 80],
                           help='lr scheduler multi-steps: List of epoch indices. Must be increasing.')
        group.add_argument("--lr-scheduler-warmup", type=int, default=10,
                           help='The number of epochs to linearly increase the learning rate.')
        group.add_argument("--stopping-learning-rate", type=float, default=1e-9,
                           help='stop when learning rate is smaller than this value')

        group = self.parser.add_argument_group('options of logging and saving')
        group.add_argument('--resume', default=None, type=str, help='path of args.yml for resuming')
        group.add_argument('--resume-checkpoint-file', default=None, type=str,
                           help='path of checkpoint file for resuming, the "checkpoints/last.checkpoint.pth" is default')
        group.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
        group.add_argument("--write-graph", action='store_true', help='visualize graph in tensorboard')
        group.add_argument('--log-images', action='store_true', help='save image samples each batch')
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        default_log_dir = os.path.join('runs', current_time + '_' + socket.gethostname())
        group.add_argument('--log-dir', type=str, default=default_log_dir, help='Location to save logs and checkpoints')
        group.add_argument('--log-loss-interval', type=int, default=0,
                           help='interval of iterations for loss logging, nonpositive for disabling')
        group.add_argument('--checkpoints-interval', type=int, default=20,
                           help='interval of epochs for saving checkpoints')

    def add_argument(self, *kargs, **kwargs):
        self.parser.add_argument(*kargs, **kwargs)

    def add_argument_group(self, *kargs, **kwargs):
        return self.parser.add_argument_group(*kargs, **kwargs)

    def set_defaults(self, **kwargs):
        self.parser.set_defaults(**kwargs)

    def parse_args(self, args=None, namespace=None):
        parsed_args = self.parser.parse_args(args=args, namespace=namespace)
        if parsed_args.resume:
            parsed_args = self.update_args_from_file(parsed_args, parsed_args.resume)
            # overwrite with command line again
            parsed_args = self.parser.parse_args(args=args, namespace=parsed_args)
            if getattr(parsed_args, 'resume_checkpoint_file', None) is None:
                # set default resume checkpoint file
                resume_dir = Path(parsed_args.resume).parent
                checkpoint_file = resume_dir / 'checkpoints' / 'last.checkpoint'
                setattr(parsed_args, 'resume_checkpoint_file', str(checkpoint_file))

        return parsed_args

    def update_args_from_file(self, args, filename):
        args_var = yaml.load(Path(filename).open())
        for name in args_var:
            if name not in ('resume_checkpoint_file'):
                setattr(args, name, args_var[name])
        return args
