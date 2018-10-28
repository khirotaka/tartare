# coding: utf-8
from tartare.Vision import DataAugmentation, MakeCategory, BuildDataset, ExpandImgData

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

MakeCategory(target_dir="apple").init(label=0, size=(64, 64), mode="RGB").export_category(filename="apple.npz", verbose=True)
MakeCategory(target_dir="melon").init(label=1, size=(64, 64), mode="RGB").export_category(filename="melon.npz", verbose=True)

BuildDataset("apple.npz", "melon.npz").export_dataset(filename="apple_melon.npz", verbose=True)

(x_train, y_train), (x_test, y_test) = \
    ExpandImgData(filename="apple_melon.npz").load_data(test_size=0.3, division=True, shuffle=True)
