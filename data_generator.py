from keras.utils import Sequence
from skimage.io import imread
import os
from sklearn.utils import shuffle
import numpy as np
from albumentations import Resize


class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir, image_folder, mask_folder,
                 batch_size=1, image_width=640, image_height=480, nb_y_features=1,
                 augmentation=None, do_shuffle=True):
        self.image_file_names = DataGeneratorFolder.listdir_full_path(os.path.join(root_dir, image_folder))
        self.mask_names = DataGeneratorFolder.listdir_full_path(os.path.join(root_dir, mask_folder))
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_width = image_width
        self.image_height = image_height
        self.nb_y_features = nb_y_features
        self.do_shuffle = do_shuffle

    @staticmethod
    def listdir_full_path(d):
        return np.sort([os.path.join(d, f) for f in os.listdir(d)])

    def __len__(self):
        return int(np.ceil(len(self.image_file_names) / self.batch_size))

    def on_epoch_end(self):
        if self.do_shuffle:
            self.image_file_names, self.mask_names = shuffle(self.image_file_names, self.mask_names)

    @staticmethod
    def read_image_mask(image_name, mask_name):
        # print(imread(image_name).astype(np.float32).shape)
        return imread(image_name).astype(np.float32) / 255, (imread(mask_name, as_gray=True) > 0).astype(np.int8)

    def __getitem__(self, index):
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_file_names)))
        indexes = self.image_file_names[data_index_min:data_index_max]
        this_batch_size = len(indexes)
        x = np.empty((this_batch_size, 480, 640, 3), dtype=np.float32)
        y = np.empty((this_batch_size, 480, 640, 1), dtype=np.uint8)
        for i, sample_index in enumerate(indexes):

            x_sample, y_sample = self.read_image_mask(self.image_file_names[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])

            if self.augmentation is not None:
                # print("sample:", x_sample.shape)
                augmented = self.augmentation()(image=x_sample, mask=y_sample)
                # image_augm = augmented['image']
                # print("augmented: ", augmented['image'].shape)
                image_augm = augmented['image'].reshape(480, 640, 3)
                mask_augm = augmented['mask'].reshape(self.image_height, self.image_width, self.nb_y_features)
                x[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
                y[i, ...] = mask_augm

            elif self.augmentation is None and self.batch_size == 1:
                x_sample, y_sample = self.read_image_mask(self.image_file_names[index * 1 + i],
                                                          self.mask_names[index * 1 + i])
                augmented = Resize(height=(x_sample.shape[0] // 32) * 32, width=(x_sample.shape[1] // 32) * 32)(
                    image=x_sample, mask=y_sample)
                x_sample, y_sample = augmented['image'], augmented['mask']

                return x_sample.reshape(1, x_sample.shape[0], x_sample.shape[1], 3).astype(
                    np.float32), y_sample.reshape(1, x_sample.shape[0], x_sample.shape[1], self.nb_y_features).astype(
                    np.uint8)

        return x, y


"""
from keras.utils import Sequence
from skimage.io import imread
import os
from sklearn.utils import shuffle
import numpy as np
from albumentations import Resize


class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir, image_folder, mask_folder,
                 batch_size=1, image_size=768, nb_y_features=1,
                 augmentation=None, do_shuffle=True):
        self.image_file_names = DataGeneratorFolder.listdir_full_path(os.path.join(root_dir, image_folder))
        self.mask_names = DataGeneratorFolder.listdir_full_path(os.path.join(root_dir, mask_folder))
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.do_shuffle = do_shuffle

    @staticmethod
    def listdir_full_path(d):
        return np.sort([os.path.join(d, f) for f in os.listdir(d)])

    def __len__(self):
        return int(np.ceil(len(self.image_file_names) / self.batch_size))

    def on_epoch_end(self):
        if self.do_shuffle:
            self.image_file_names, self.mask_names = shuffle(self.image_file_names, self.mask_names)

    @staticmethod
    def read_image_mask(image_name, mask_name):
        return imread(image_name).astype(np.float32) / 255, (imread(mask_name, as_gray=True) > 0).astype(np.int8)

    def __getitem__(self, index):
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_file_names)))
        indexes = self.image_file_names[data_index_min:data_index_max]
        this_batch_size = len(indexes)
        x = np.empty((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):

            x_sample, y_sample = self.read_image_mask(self.image_file_names[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])

            if self.augmentation is not None:

                augmented = self.augmentation(self.image_size)(image=x_sample, mask=y_sample)
                image_augm = augmented['image']
                mask_augm = augmented['mask'].reshape(self.image_size, self.image_size, self.nb_y_features)
                x[i, ...] = np.clip(image_augm, a_min=0, a_max=1)
                y[i, ...] = mask_augm

            elif self.augmentation is None and self.batch_size == 1:
                x_sample, y_sample = self.read_image_mask(self.image_file_names[index * 1 + i],
                                                          self.mask_names[index * 1 + i])
                augmented = Resize(height=(x_sample.shape[0] // 32) * 32, width=(x_sample.shape[1] // 32) * 32)(
                    image=x_sample, mask=y_sample)
                x_sample, y_sample = augmented['image'], augmented['mask']

                return x_sample.reshape(1, x_sample.shape[0], x_sample.shape[1], 3).astype(
                    np.float32), y_sample.reshape(1, x_sample.shape[0], x_sample.shape[1], self.nb_y_features).astype(
                    np.uint8)

        return x, y

"""
