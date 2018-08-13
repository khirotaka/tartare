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
