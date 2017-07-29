from __future__ import print_function
import os
import struct
import lmdb
import caffe
import numpy as np
import logging
from PIL import Image


logging.basicConfig(level=logging.INFO)


# TODO: put a batch size here
def decode(images_path, labels_path):
    """
    Reads data from images_path, returns generator that
    yields image data and label.
    """
    images_path = os.path.expanduser(images_path)
    labels_path = os.path.expanduser(labels_path)
    with open(images_path, 'rb') as img_file:
        with open(labels_path, 'rb') as label_file:
            bs = img_file.read(16)
            # the '>' means use standard size, not current machine size
            x = struct.unpack('>iiii', bs)
            (_, num_images, num_rows, num_columns) = x
            image_size = num_rows * num_columns
            label_file.read(8)

            for n in xrange(num_images):
                bs = img_file.read(image_size)
                label = struct.unpack('>B', label_file.read(1))[0]
                img_arr = np.fromstring(bs, dtype=np.uint8)
                # image array contains ints 0 - 255
                img_arr = img_arr.astype('uint8').reshape((28, 28))
                yield (img_arr, label)


def write_image(img_arr, out_path):
    out_path = os.path.expanduser(out_path)
    Image.fromarray(img_arr, mode='L').save(out_path, 'jpeg')


def write_lmdb(gen, lmdb_path):
    """
    Write image and label to an lmdb file.
    """
    lmdb_path = os.path.expanduser(lmdb_path)
    img_db = lmdb.open(lmdb_path, map_size=int(1e9))
    with img_db.begin(write=True) as lmdb_txn:
        for (i, pair) in enumerate(gen):
            idx = '{:0>10d}'.format(i)
            logging.info('Adding {} with label: {}'.format(idx, pair[1]))

            # reshape the image
            img = pair[0]
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img = np.transpose(img, (2,0,1))

            # serialize and write
            img_datum = caffe.io.array_to_datum(img)
            img_datum.label = pair[1]
            lmdb_txn.put(idx, img_datum.SerializeToString())

    img_db.close()


if __name__ == '__main__':
    images_path = '~/mnist/data/train-images-idx3-ubyte'
    labels_path = '~/mnist/data/train-labels-idx1-ubyte'
    train_lmdb_path = '~/mnist/data/train_lmdb'

    test_images_path = '~/mnist/data/t10k-images-idx3-ubyte'
    test_labels_path = '~/mnist/data/t10k-labels-idx1-ubyte'
    test_lmdb_path = '~/mnist/data/test_lmdb'

    gen = decode(images_path, labels_path)
    write_lmdb(gen, train_lmdb_path)
    gen_ = decode(test_images_path, test_labels_path)
    write_lmdb(gen_, test_lmdb_path)
    logging.info('Success')
