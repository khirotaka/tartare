# coding: utf-8

# MIT License
#
# Copyright (c) 2018 Hirotaka Kawashima
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import os
import sys
import numpy as np
from PIL.JpegImagePlugin import JpegImageFile
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split


class DataAugmentation(object):
    """DataAugmentation class.
    Give the directory name containing the image you want to increase as argument.

    """
    def __init__(self, target_dir):
        """Initializer for DataAugmentation.

        :param target_dir: String
            Target directory name
        """
        self.__current_path = os.getcwd()                                   # カレントディレクトリを取得
        self.__target_path = self.__current_path + "/" + target_dir
        self.__file_path = []
        self.__file_name = []

    def init(self):                                               # 対象ディレクトリ内の画像ファイルの名前を全部取得
        """Gets all names of image files in the specified directory.

        :return: object
        """
        for filename in os.listdir(self.__target_path):
            if os.path.isfile(self.__target_path + "/" + filename):
                self.__file_path.append(self.__target_path + "/" + filename)   # 画像へのパスを格納
                self.__file_name.append(filename)
        return self

    @staticmethod
    def __mirror(image):
        """Flips the specified image horizontally.

        :param image:
        :return:
        """
        image = np.array(image)
        image = image[:, ::-1, :]
        image = Image.fromarray(image)
        return image

    @staticmethod
    def __flip(image):
        """Flip the specified image vertically.

        :param image:
        :return:
        """
        image = np.array(image)
        image = image[::-1, :, :]
        image = Image.fromarray(image)
        return image

    @staticmethod
    def __random_brightness(image):
        """Decrease the brightness of the specified image randomly.

        :param image:
        :return:
        """
        image = ImageEnhance.Brightness(image)
        image = image.enhance(np.random.uniform(low=0.5, high=0.8))
        return image

    @staticmethod
    def __random_contrast(image):
        """Raise the contrast of the specified image randomly.

        :param image:
        :return:
        """
        image = ImageEnhance.Contrast(image)
        image = image.enhance(np.random.uniform(low=0.3, high=0.8))
        return image

    @staticmethod
    def _random_masked(image):
        """Based on the vertical or horizontal length of the input image,
        make a mask 0.3 - 0.5 times the length of the shorter side, and mask the random position of the image.

        Reference: Improved Regularization of Convolutional Neural Networks with Cutout

        :param image:

        :return:
        """
        image = np.array(image)
        mask_value = image.mean()
        h, w, _ = image.shape

        mask_size = np.random.uniform(0.3, 0.5) * w
        if h < w:
            mask_size = np.random.uniform(0.3, 0.5) * h

        top = np.random.randint(0 - mask_size // 2, h - mask_size)
        left = np.random.randint(0 - mask_size // 2, w - mask_size)

        bottom = int(top + mask_size)
        right = int(left + mask_size)

        if top < 0:
            top = 0
        if left < 0:
            left = 0
        masked_image = np.copy(image)
        masked_image[top:bottom, left:right, :].fill(mask_value)
        masked_image = Image.fromarray(masked_image)
        return masked_image

    def __write_image(self, input_image, file_name):
        """ Store the created image in the same directory as the original image.

        :param input_image:

        :param file_name:

        :return: Image file
        """
        try:
            input_image.save(self.__target_path + "/" + file_name)

        except OSError:
            print("*"*20)
            print("ERROR: {}".format(OSError))
            print("*" * 20)

        except MemoryError:
            print("*" * 20)
            print("ERROR: {}".format(MemoryError))
            print("*" * 20)

    def augment(self, mirror=False, flip=False, brightness=False, contrast=False, mask=False):
        """Effect is applied to the image based on the argument, and it saves as an image file.

        :param mirror: Bool
            Default: False

        :param flip:
            Default: False

        :param brightness:
            Default: False

        :param contrast:
            Default: False

        :param mask:
            Default: False

        :return: Image files
        """
        for i, file in enumerate(self.__file_path):
            img = Image.open(file)

            if mirror:
                mirrored_img = self.__mirror(img)
                self.__write_image(mirrored_img, "/mirrored_{}".format(self.__file_name[i]))

            if flip:
                flipped_img = self.__flip(img)
                self.__write_image(flipped_img, "/flipped_{}".format(self.__file_name[i]))

            if brightness:
                brightened_img = self.__random_brightness(img)
                self.__write_image(brightened_img, "/brightened_{}".format(self.__file_name[i]))

            if contrast:
                edited_img = self.__random_contrast(img)
                self.__write_image(edited_img, "/contrasted_{}".format(self.__file_name[i]))

            if mask:
                for j in range(5):
                    masked_img = self._random_masked(img)
                    self.__write_image(masked_img, "/masked_{}_{}".format(j, self.__file_name[i]))


class MakeCategory(object):
    """Create an .npz file which containing image arrays and labels.

    If you want to create a data-set with many categories,
    create a directory containing images as many as the number of categories,
    convert each directory to an .npz file, and use the class `BuildDataset()` to generate a data-set.
    """
    def __init__(self, target_dir):
        """Initializer for CreateCategory.

        :param target_dir: String
            Specify the directory name where the image is saved.
        """
        self.__current_path = os.getcwd()                                   # カレントディレクトリを取得
        self.__target_path = self.__current_path + "/" + target_dir
        self.__file_names = []
        self.__image_files = []
        self.__labels = []

    def __get_image_name(self):                                               # 対象ディレクトリ内の画像ファイルの名前を全部取得
        """Gets all names of image files in the specified directory.

        :return: object
        """
        for filename in os.listdir(self.__target_path):
            _, fmt = os.path.splitext(filename)
            if fmt == ".jpg" or fmt == ".JPG" or fmt == ".jpeg" or fmt == ".JPEG":
                if os.path.isfile(self.__target_path + "/" + filename):
                    self.__file_names.append(self.__target_path + "/" + filename)   # 画像へのパスを格納
            else:
                sys.stderr.write("ERROR: Contained unsupported file format. This version only supports JPEG format.\n")
                sys.stderr.write("Delete files other than JPEG format.\n")
                break
        return self

    def __read_image(self, size=(64, 64)):                                        # 画像ファイルをnumpy配列にして、それを配列に格納
        """Convert the image in the specified directory to NumPy array
         based on the file name obtained by the `__get_image_name()` method.

        :return: object
        """
        for file_name in self.__file_names:
            img = Image.open(file_name)
            if type(img) is JpegImageFile:
                img = img.resize(size)
                img = np.array(img)
                self.__image_files.append(img)
            else:
                sys.stderr.write("ERROR: Unsupported file format. This version only supports JPEG format.\n")
                break
        return self

    def __make_label(self, label=0):
        """Create a label for supervised learning. Must be Unsigned Integer !

        :param label: Int
            The label to assign to the images. Must be Unsigned Integer !

        :return:
        """
        if type(label) is int:
            for i in range(len(self.__file_names)):
                self.__labels.append(label)
            return self
        else:
            print("Error: The value assigned to the label variable must be `Positive Integer`.")

    def init(self, label, size):
        """Convert the image in the target directory to NumPy array and assign an appropriate label.

        :param label: Int
            The label to assign to the images. Must be Unsigned Integer!

        :param size: (Int, Int)


        :return:
        """
        self.__get_image_name().__read_image(size=size).__make_label(label=label)
        return self

    def export_category(self, filename, verbose=False):                                 # ファイルの書きだし、ファイル名のみで良い。.npzは不要
        """Export the .npz file based on the data stored in the array.

        :param filename: String
            Name of the .npz file.

        :param verbose: Bool
            If True, display the log. Logs are output even if False when an error occurs.

        :return:
            .npz file
        """
        np_labels = np.array(self.__labels, dtype=np.uint8)
        np_labels = np_labels.reshape(np_labels.shape[0], 1)            # 更新
        np_image_files = np.array(self.__image_files, dtype=np.uint8)
        try:
            np.savez_compressed(filename, image=np_image_files, label=np_labels)

        except OSError:
            print("*" * 20)
            print("ERROR: {}".format(OSError))
            print("*" * 20)

        except MemoryError:
            print("*" * 20)
            print("ERROR: {}".format(MemoryError))
            print("*" * 20)

        else:
            if verbose:
                print("file name: {}".format(filename))


class BuildDataset(object):
    """Create a data-set based on the .npz file containing images and labels created with `CreateCategory()`.

    """
    def __init__(self, *args):
        """It reads by specifying the .npz file created by `CreateCategory()` and stores it in the array.

        :param args: String
            arg[1] : file name 1 (.npz)
            arg[2] : file name 2 (.npz)
            ...

        """
        load_file = np.load(args[0])
        img_data = load_file["image"]
        label_data = load_file["label"]

        self.__img = np.copy(img_data)
        self.__label = np.copy(label_data)

        for i in range(1, len(args)):
            load_file = np.load(args[i])

            img_data = load_file["image"]
            label_data = load_file["label"]

            self.__img = np.append(self.__img, img_data, axis=0)
            self.__label = np.append(self.__label, label_data, axis=0)

    def export_dataset(self, filename, verbose=False):
        """Export the data-set based on the data stored in the array at instance creation time.

        :param filename: String
            Name of the date set.

        :param verbose: Bool
            If True, display the log. Logs are output even if False when an error occurs.

        :return:
            .npz file

        """
        np_imgs = np.array(self.__img, dtype=np.uint8)
        np_labels = np.array(self.__label, dtype=np.uint8)

        try:
            np.savez_compressed(filename, image=np_imgs, label=np_labels)

        except OSError:
            print("*" * 20)
            print("ERROR: {}".format(OSError))
            print("*" * 20)

        except MemoryError:
            print("*" * 20)
            print("ERROR: {}".format(MemoryError))
            print("*" * 20)

        else:
            if verbose:
                print("Data set name: {}".format(filename))


class ExpandImgData(object):
    """Expands the specified data set and extracts image data and label.

    """
    def __init__(self, filename):
        """Specify the data set to be expanded.

        :param filename: String
            Target file.
        """
        self.__loaded_file = np.load(filename)
        self.__images = self.__loaded_file["image"]
        self.__labels = self.__loaded_file["label"]

    def load_data(self, test_size=0.3, division=True, shuffle=True):
        """Expands the data set, extracts image data and labels, returns lists or tuples based on arguments.

        :param test_size: Float (default=0.3)
            Percentage of test size. (It should be 0.0 ~ 1.0.)
            Set the ratio of the number of test samples.

        :param division: Bool
            Whether to split the data into batches for training and testing.

        :param shuffle: Bool
            Whether to shuffle the data before splitting into batches.

        :return: List ot Tuple
            IF division == True
                Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
            IF division == False
                List of Numpy arrays: `x, y`.
        """
        try:
            if division:
                x_train, x_test, y_train, y_test = \
                    train_test_split(self.__images, self.__labels, test_size=test_size, shuffle=shuffle)
                return (x_train, y_train), (x_test, y_test)

            else:
                x = self.__images
                y = self.__labels

                return x, y
        except MemoryError:
            print("*" * 20)
            print("ERROR: {}".format(MemoryError))
            print("*" * 20)
