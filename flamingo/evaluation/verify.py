import argparse
import os
import torch
# import ..models.face_resnet as models 
from .verification.data_reader import load_bin
from .verification.verification import ver_test
import torchvision.transforms as transforms


class VerificationRunner(object):
    def __init__(self, model, datasets: list, batch_size: int):
        # load verification data
        ver_list = []
        ver_name_list = []
        self.metrics = {}
        transformers = []
        for flip in [False, True]:
            transformers.append(transforms.Compose([
                                transforms.RandomHorizontalFlip(1 if flip else 0),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                     std=[1, 1, 1])]))

        for db_path in datasets:
            data_set = load_bin(db_path, [112, 112], transformers)
            ver_list.append(data_set)
            _, db = os.path.split(db_path)
            print('begin db %s convert.' % db)
            ver_name_list.append(db)
            self.metrics[db] = {'xnorm': [],
                                'acc': [],
                                'std': []}

        self.ver_list = ver_list
        self.ver_name_list = ver_name_list
        self.batch_size = batch_size
        self.model = model

    def run(self):
        self.model.eval()
        with torch.no_grad():
            for db, xnorm, acc, std in ver_test(self.ver_list, self.ver_name_list, self.model, self.batch_size):
                # _, name = os.path.split(path)
                self.metrics[db]['xnorm'] = xnorm
                self.metrics[db]['acc'] = acc
                self.metrics[db]['std'] = std


if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet100',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="checkpoint path for verification")
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--experiment', help='tag of the experiment', default='DefaultModel')
    parser.add_argument('--type', help='type of evaluation', choices=['Verification', 'Identification'], default='Verification')
    args = parser.parse_args()

    print("=> loading checkpoint '{}'".format(args.checkpoint))
    model = models.__dict__[args.arch]()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        if args.gpu is not None:
            model = model.cuda(args.gpu)
        else:
            model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint.state_dict())

    if args.type == 'Verification':

        runner = VerificationRunner(model,
                                    ['./lfw.bin',
                                     './verification/cfp_ff.bin',
                                     './verification/cfp_fp.bin',
                                     './verification/agedb_30.bin'],
                                    args.batch_size)

    runner.run()
