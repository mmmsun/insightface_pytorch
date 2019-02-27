import lmdb
import os
from tensorpack.dataflow.serialize import dumps, get_tqdm, _reset_df_and_get_size, logger, DataFlow


class LMDBSerializer(object):
    """
    Serialize a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.

    You will need to `pip install lmdb` to use it.
    """
    @staticmethod
    def save(df, path, write_frequency=5000):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output path. Either a directory or an lmdb file.
            write_frequency (int): the frequency to write back data to disk.
        """
        assert isinstance(df, DataFlow), type(df)
        isdir = os.path.isdir(path)
        if isdir:
            assert not os.path.isfile(os.path.join(path, 'data.mdb')), "LMDB file exists!"
        else:
            assert not os.path.isfile(path), "LMDB file {} exists!".format(path)
        db = lmdb.open(path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True)    # need sync() at the end
        size = _reset_df_and_get_size(df)
        with get_tqdm(total=size) as pbar:
            idx = -1

            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            txn = db.begin(write=True)
            for idx, dp in enumerate(df):
                txn.put(u'{:08}'.format(idx).encode('ascii'), dumps(dp))
                pbar.update()
                if (idx + 1) % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
            txn.commit()

            keys = [u'{:08}'.format(k).encode('ascii') for k in range(idx + 1)]
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps(keys))
            
            classes = [u'{:08}'.format(k).encode('ascii') for k in df.classes]
            with db.begin(write=True) as txn:
                txn.put(b'__classes__', dumps(classes))

            logger.info("Flushing database ...")
            db.sync()
        db.close()