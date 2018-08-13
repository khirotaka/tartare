Tartare: Make homebrew image dataset for machine learning.
---

## You can create your own dataset for Keras and other.

It is a library to easily create a data set for Keras based on the **JPEG** images you collected.
In addition to creating the data set, there is also a data augmentation function.


This version only supports JPEG format.

Compatible with: Python 3.5, 3.6

Operation confirmed:
 - macOS 10.13 High Sierra
 - Ubuntu 17.10 Artful Aardvark & Ubuntu 18.04 Bionic Beaver

## Requirement
- NumPy 1.14.5
- SciPy >= 1.1.0
- Pillow 5.2.0
- Scikit-Learn 0.19.2

## Usage
Precondition
 - Directory Strucure
 
 ```
.
├── tutorial.py
└── apple/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── 4.jpg
│  
└── melon/
    ├── 0.jpg
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    └── 4.jpg
```


### 1, Data Augmentation

You can augment your images.

- `mirror=True` : Flips the specified image horizontally.
- `flip=True` : Flip the specified image vertically.
- `brightness=True` : Decrease the brightness of the specified image randomly.
- `contrast=True` : Raise the contrast of the specified image randomly.
- `mask=True` : Based on the vertical or horizontal length of the input image,
make a mask 0.3 - 0.5 times the length of the shorter side, and mask the random position of the image.
Create 5 pictures.


`tutorial.py`
---
```
# coding: utf-8
from tartare.Vision import DataAugmentation

DataAugmentation("apple").init().augment(mirror=True,
                                         flip=True,
                                         brightness=True,
                                         contrast=True,
                                         mask=True)

DataAugmentation("melon").init().augment(mirror=True,
                                         flip=True,
                                         brightness=True,
                                         contrast=True,
                                         mask=True)
```



### 2, MakeCategory

You can receive `.npz` file with contained image data & label 

**Caution:** When creating an `.npz` file with MakeCategory (), 
make sure that the number of original image files is the same for all categories.

**f you are Mac user, try to remove `.DS_Store` file.**


- `target_dir` : Directory name where images of categories are saved.
- `label` : Correct answer label for supervised learning.
- `size` : Image size when saving. Tuple. (width, height).
- `filename`: File name when output.
- `verbose` : When True is selected, when the file output succeeds, the result is output.

`tutorial.py`
---
```
from tartare.Vision import MakeCategory

MakeCategory(target_dir="apple").init(label=0, size=(64, 64)).export_category(filename="apple.npz", verbose=True)
MakeCategory(target_dir="melon").init(label=1, size=(64, 64)).export_category(filename="melon.npz", verbose=True)

```

### 3, BuildDataset

Create a data set based on the .npz file created with MakeCategory ().

**Caution:** When creating an `.npz` file with MakeCategory (), 
make sure that the number of original image files is the same for all categories. 

- `BuildDataset(filename1, filename2, ...)` : Specify `.npz` for each category.
- `filename`: File name when output.
- `verbose` : When True is selected, when the file output succeeds, the result is output.

`tutorial.py`
---
```
from tartare.Vision import BuildDataset

BuildDataset("apple.npz", "melon.npz").export_dataset(filename="apple_melon.npz",verbose=True)
```

### 4, ExpandImgData

The image (tensor) and label (two dimensions) are given as return values.

- `filename=` Name of Dataset.
- `test_size`: the proportion of the dataset to include in the test split. Default: 0.3
- `division`: Whether or not to divide the data set for training and testing. (Default=True)
- `shuffle`: Whether to shuffle the data set. (Default=True)


`tutorial.py`
---
```
from tartare.Vision import ExpandImgData

(x_train, y_train), (x_test, y_test) = ExpandImgData(filename="apple_melon.npz").load_data(test_size=0.3,
                                                                                      division=True,
                                                                                      shuffle=True)
```

### 5, Sample
 
 
`Sample_NN.py`
---
```
# coding: utf-8
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from tartare.Vision import ExpandImgData

def main():
    (x_train, y_train), (x_test, y_test) = ExpandImgData("apple_melon.npz").load_data(test_size=0.3,
                                                                                      division=True,
                                                                                      shuffle=True)

    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    input_shape = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]

    x_train = x_train.reshape(x_train.shape[0], input_shape)  # 2次元配列を1次元に変換
    x_test = x_test.reshape(x_test.shape[0], input_shape)

    train_image = x_train.astype("float32")
    test_image = x_test.astype("float32")

    train_image /= 255.0
    test_image /= 255.0

    train_label = np_utils.to_categorical(y=y_train, num_classes=2)
    test_label = np_utils.to_categorical(y=y_test, num_classes=2)


    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(input_shape,)))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    epoch_num = 20
    history = model.fit(x=train_image, y=train_label, batch_size=4, epochs=epoch_num, validation_split=0.1, verbose=1)

    model.summary()

    score = model.evaluate(x=test_image, y=test_label, verbose=0)
    print("Test Loss: {}".format(score[0]))
    print("Test Accuracy: {}".format(score[1]))

if __name__ == "__main__":
    main()

```


## Install
- Install Tartare from PyPI:

    `pip3 install tartare`

## License
MIT

## Author

Hirotaka Kawashima

