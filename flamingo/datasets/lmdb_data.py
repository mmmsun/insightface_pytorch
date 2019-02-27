import torch.utils.data as data
from PIL import Image
import os
import os.path
import lmdb
from tensorpack.utils.compatible_serialize import loads
import io


class LMDBDataset(data.Dataset):
    """
    Modified from https://tensorpack.readthedocs.io/modules/dataflow.html#tensorpack.dataflow.LMDBData
    Read a LMDB database and produce (k,v) raw bytes pairs.
    The raw bytes are usually not what you're interested in.
    """
    def __init__(self, lmdb_path,  transform=None, target_transform=None):

        self._lmdb_path = lmdb_path
        # self._shuffle = shuffle

        self._open_lmdb()
        self._size = self._txn.stat()['entries']
        self._set_keys()
        # logger.info("Found {} entries in {}".format(self._size, self._lmdb_path))
        # self._guard = DataFlowReentrantGuard()
        self.transform = transform
        self.target_transform = target_transform

    def _set_keys(self):
        self.keys = self._txn.get(b'__keys__')
        self.keys = loads(self.keys)
        self.classes = self._txn.get(b'__classes__')
        self.classes = loads(self.classes)

    def _open_lmdb(self):
        self._lmdb = lmdb.open(self._lmdb_path,
                               subdir=os.path.isdir(self._lmdb_path),
                               readonly=True, lock=False, readahead=True,
                               map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()

    def reset_state(self):
        self._lmdb.close()
        super(LMDBDataset, self).reset_state()
        self._open_lmdb()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        sample = loads(self._txn.get(self.keys[index]))
        img = sample[0]
        label = sample[1]
        encoded_jpg_io = io.BytesIO(img)
        image = Image.open(encoded_jpg_io)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (image, label)
