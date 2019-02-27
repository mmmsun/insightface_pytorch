import torch
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .lmdb_data import LMDBDataset


def get_data_loaders(path: str, batch_size: int, num_workers: int,
                     distributed: bool=False, transform: transforms=None, 
                     data_type: str='imagefolder'):
    if transform is None:
        transform = transforms.Compose([
                    transforms.RandomResizedCrop(112, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[1, 1, 1])])
    if data_type == 'lmdb':
        dataset = LMDBDataset(path, transform)
    elif data_type == 'imagefolder':
        dataset = datasets.ImageFolder(path, transform)
    else:
        raise ValueError('not valid data_type: {}'.format(data_type))
    num_classes = len(dataset.classes)
    print('num_classes', num_classes)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    return dataloader, num_classes
