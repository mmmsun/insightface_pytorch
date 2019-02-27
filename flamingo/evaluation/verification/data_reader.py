import numpy as np
import pickle
import io
import PIL.Image


def load_bin(db_path: str, image_size: list, transformers):
    # compatible to python2 https://docs.python.org/3/library/pickle.html#pickle.load
    bins, issame_list = pickle.load(open(db_path, 'rb'), encoding='bytes', fix_imports=True)
    data_list = []
    for _ in [0, 1]:
        data = np.empty((len(issame_list)*2, 3, image_size[0], image_size[1]))
        # data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        encoded_jpg_io = io.BytesIO(_bin)
        image = PIL.Image.open(encoded_jpg_io)
        img = np.array(image)

        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('test', img)
        # cv2.waitKey(0)
        for flip, transform in enumerate(transformers):
            imgpil = PIL.Image.fromarray(img)
            input_data = transform(imgpil)
            data_list[flip][i, ...] = input_data  # img2.astype(np.float32)
            # print('data_list', i, data_list[flip].dtype, data_list[flip].shape)

        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    # print('data_list', data_list[0].max(), data_list[0].min())
    return data_list, issame_list
