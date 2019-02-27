from .dump_feature import save_features
import torch
from ...datasets.lmdb_data import LMDBDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from ...utils.utils import load_json

dir_path = os.path.dirname(os.path.realpath(__file__))


class MegafaceEvaluationRunner(object):
    def __init__(self, model, batch_size, num_workers,
                 megaface_lmdb_path, facescrub_lmdb_path,
                 dump_dir,
                 size=100000,
                 mf_noiselist=os.path.join(dir_path, 'megaface_noises.txt'),
                 fs_noiselist=os.path.join(dir_path, 'facescrub_noises.txt'),
                 feature_ext=1):
        # load verification data
        self.metrics = {}
        self.megaface_dataloaders = self.get_dataloader(megaface_lmdb_path, batch_size, num_workers)
        self.facescrub_dataloaders = self.get_dataloader(facescrub_lmdb_path, batch_size, num_workers)
        self.batch_size = batch_size
        self.model = model
        self.dump_dir = dump_dir
        self.size = size
        self.feature_ext = feature_ext
        self.mf_noise_map = {}
        for line in open(mf_noiselist, 'r'):
            if line.startswith('#'):
                continue
            line = line.strip()
            _vec = line.split("\t")
            if len(_vec) > 1:
                line = _vec[1]
            self.mf_noise_map[line] = 1

        self.fs_noise_map = {}
        for line in open(fs_noiselist, 'r'):
            if line.startswith('#'):
                continue
            line = line.strip()
            fname = line.split('.')[0]
            p = fname.rfind('_')
            fname = fname[0:p]
            self.fs_noise_map[line] = fname
        self.metrics = {'acc': []}

    @staticmethod
    def get_dataloader(path, batch_size, num_workers):
        dataloaders = []
        for flip in [False, True]:
            transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(1 if flip else 0),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[1, 1, 1])])
            dataset = LMDBDataset(path, transform)

            dataloaders.append(DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True, drop_last=False))

        return dataloaders

    def run(self, iteration):
        save_dir = os.path.join(self.dump_dir, '{:012d}'.format(iteration))
        self.model.eval()
        with torch.no_grad():
            # dump features to local disk
            # facescrub
            save_features(self.model, self.facescrub_dataloaders[0], self.facescrub_dataloaders[1],
                          save_dir, 'facescrub', self.feature_ext, fs_noise_map=self.fs_noise_map,
                          mf_noise_map=None, save_original=False)

            # megaface
            save_features(self.model, self.megaface_dataloaders[0], self.megaface_dataloaders[1],
                          save_dir, 'megaface', self.feature_ext, fs_noise_map=None,
                          mf_noise_map=self.mf_noise_map, save_original=False)
            command = 'touch {}'.format(save_dir, 'finished-{}'.format(self.size))
            os.system(command)
