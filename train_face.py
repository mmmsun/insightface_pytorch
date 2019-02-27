"""
 Requirements:
    TensorboardX (https://github.com/lanpa/tensorboard-pytorch): `pip install tensorboardX`
    Tensorboard: `pip install tensorflow` (or just install tensorboard without the rest of tensorflow)
 Usage:
    Start tensorboard:
    ```bash
    tensorboard --logdir=/tmp/tensorboard_logs/
    ```
    Run the example:
    ```bash
    python facenet_with_tensorboardx.py --log_dir=/tmp/tensorboard_logs
    ```
"""

from __future__ import print_function

import random
import warnings
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

import flamingo.losses.loss as loss
import flamingo.models.classifier as classifier
import flamingo.models.embeddings as embeddings
import flamingo.models.face_resnet as models
import flamingo.optim.optim as optim
import flamingo.optim.lr_scheduler as lr_scheduler
from flamingo import create_supervised_trainer
from flamingo.datasets.load import get_data_loaders
from flamingo.evaluation.megaface.evaluate import MegafaceEvaluationRunner
from flamingo.evaluation.verify import VerificationRunner
from flamingo.metrics.accuracy import AccuracyIter
from flamingo.utils.utils import create_summary_writer, load_yaml
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint


def main(args):
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    config = load_yaml(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, num_classes = get_data_loaders(num_workers=args.workers,
                                                 distributed=args.distributed,
                                                 **config['data'])

    arch_config = config['arch']['param']
    arch_config['embedding_block'] = embeddings.__dict__[config['arch']['embedding_block']['name']]\
        (**config['arch']['embedding_block']['param'])
    arch_config['classifier_block'] = classifier.__dict__[config['arch']['classifier_block']['name']]\
        (num_embedding=config['arch']['embedding_block']['param']['num_embedding'], num_class=num_classes)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(config['arch']['name']))
        model = models.__dict__[config['arch']['name']](pretrained=True, **arch_config)
    else:
        print("=> creating model '{}'".format(config['arch']['name']))

        model = models.__dict__[config['arch']['name']](**arch_config)

    criterion = loss.__dict__[config['loss']['name']](**config['loss']['param'])

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    print('model', model)
    model.to(device)

    writer = create_summary_writer(model, train_loader, config['log_dir'])

    optimizer = optim.__dict__[config['optimizer']['name']](model.parameters(), **config['optimizer']['param'])

    scheduler = lr_scheduler.__dict__[config['lr_scheduler']['name']](optimizer, **config['lr_scheduler']['param'])
    trainer = create_supervised_trainer(model, optimizer, criterion, scheduler, device=device)

    def output_transform(output):
        y_pred = output[2]
        y = output[3]
        return y_pred, y

    metric = AccuracyIter(output_transform=output_transform)
    metric.attach(trainer, "accuracy")

    if args.prefix == "":
        prefix = config['arch']['name']
    checkpoint_handler = ModelCheckpoint(config['checkpoint_dir'], prefix,
                                         n_saved=200,
                                         save_interval=1,
                                         create_dir=True,
                                         require_empty=False)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'model': model, 'optimizer': optimizer})
    if torch.cuda.device_count() > 1:
        evaluator_face = VerificationRunner(torch.nn.DataParallel(model.module.net), config['verification']['dataset'],
                                            config['data']['batch_size'])
        if 'megaface_evaluation' in config:
            evaluator_megaface = MegafaceEvaluationRunner(torch.nn.DataParallel(model.module.net), config['data']['batch_size'],
                                                          args.workers, prefix, **config['megaface_evaluation'])

    else:
        evaluator_face = VerificationRunner(model.net, config['verification']['dataset'], config['data']['batch_size'])
        if 'megaface_evaluation' in config:
            evaluator_megaface = MegafaceEvaluationRunner(model.net, config['data']['batch_size'], args.workers, prefix,
                                                          **config['megaface_evaluation'])


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % config['log_interval'] == 0:
            print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f} Accuracy: {:.5f}"
                  "".format(engine.state.epoch, iter, len(train_loader), engine.state.output[0],
                            engine.state.metrics['accuracy']))
            writer.add_scalar("training/loss", engine.state.output[0], engine.state.iteration)
            writer.add_scalar("training/accuracy", engine.state.metrics['accuracy'], engine.state.iteration)
            writer.add_scalar("training/lr", optimizer.param_groups[0]['lr'], engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_scheduler(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_validation_results(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if engine.state.iteration % int(len(train_loader) / args.eval_interval) == 0:
            evaluator_face.run()
            for db in evaluator_face.metrics:
                writer.add_scalar("verfication/{}/xnorm".format(db), evaluator_face.metrics[db]['xnorm'], engine.state.iteration)
                writer.add_scalar("verfication/{}/accuracy".format(db), evaluator_face.metrics[db]['acc'], engine.state.iteration)
                writer.add_scalar("verfication/{}/std".format(db), evaluator_face.metrics[db]['std'], engine.state.iteration)
                print("Verfication Results {} - Epoch[{}] Iteration[{}/{}]  xnorm: {:.4f} acc: {:.5f} std: {:.5f}"
                      .format(db, engine.state.epoch, iter, len(train_loader), evaluator_face.metrics[db]['xnorm'],
                              evaluator_face.metrics[db]['acc'], evaluator_face.metrics[db]['std']))
            checkpoint_handler(engine, {'model': model, 'optimizer': optimizer, 'verification': evaluator_face.metrics})
            if 'megaface_evaluation' in config:
                # evaluator_megaface.run(engine.state.iteration)
                writer.add_scalar("megaface/{}/accuracy".format(evaluator_megaface.size), evaluator_megaface.metrics['acc'], engine.state.iteration)
                print("Megaface Results {} - Epoch[{}] Iteration[{}/{}] acc: {:.5f}"
                      .format(evaluator_megaface.size, engine.state.epoch, iter, len(train_loader), evaluator_face.metrics['acc']))

    # kick everything off
    trainer.run(train_loader, max_epochs=config['epochs'])

    writer.close()


if __name__ == "__main__":

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    embedding_names = sorted(name for name in embeddings.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(embeddings.__dict__[name]))

    parser = ArgumentParser()
    parser.add_argument('config', metavar='CFG',
                        help='path to training & model configuration')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--eval_interval', type=float, default=10,
                        help='how many times evaluation in one epoch')
    parser.add_argument("--prefix", type=str, default="",
                        help="prefix of checkpoints")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    args = parser.parse_args()

    main(args)
