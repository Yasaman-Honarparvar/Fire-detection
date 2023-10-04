import imageio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from PIL import Image
import sys


def main():
    
    image_to_dataset()
    fire_detection()
    show_pred_fire()


def show_pred_fire():
    """
    It is a function that highlights the fire based on the trained dataset and save the result into 'myimg.jpeg'
    """
    df = pd.read_csv(
        "pred_dataset.txt",
        sep=" ",
        usecols=[0, 1, 2, 3],
        names=["red", "green", "blue", "label"],
    )
    red = df["red"]
    rslt_df = df[df["label"] == 1]
    a = []
    a = list(rslt_df.index)
    pic = imageio.imread("firewater2.jpg")
    for i in range(0, len(a) - 1):
        if (
            a[i] + 1 == a[i + 1]
            and df.iloc[a[i], [0]].values > 160
            and df.iloc[a[i], [1]].values > 80
            and df.iloc[a[i], [2]].values > 0
        ):
            df.iloc[a[i], [0]] = 255
            df.iloc[a[i], [1]] = 0
            df.iloc[a[i], [2]] = 0
    image_r = np.float64(df.iloc[:, 0])
    image_g = np.float64(df.iloc[:, 1])
    image_b = np.float64(df.iloc[:, 2])
    arr_red = np.zeros((pic.shape[0] + 1, pic.shape[1]))
    arr_blue = np.zeros((pic.shape[0] + 1, pic.shape[1]))
    arr_green = np.zeros((pic.shape[0] + 1, pic.shape[1]))
    counter = 0
    for i in range(0, len(image_r)):
        arr_red[counter][i % (pic.shape[1])] = image_r[i]
        arr_blue[counter][i % (pic.shape[1])] = image_b[i]
        arr_green[counter][i % (pic.shape[1])] = image_g[i]
        if i % (pic.shape[1]) == 0:
            counter = counter + 1
    rgbArray = np.zeros((pic.shape[0] + 1, pic.shape[1], 3), "uint8")
    rgbArray[..., 0] = arr_red
    rgbArray[..., 1] = arr_green
    rgbArray[..., 2] = arr_blue
    img = Image.fromarray(rgbArray)
    img.save("myimg.jpeg")
    plt.imshow(img)


def fire_detection():
    """
    It is a function that trains dataset (image_dataset.txt) and get an image
    as input to predict the label of the image.
    The result of this prediction that results in labeling the image is saved in the pred_dataset.txt
    """

    f1 = df_fire_d("firewater2.jpg")
    data = pd.read_csv(
        "image_dataset.txt",
        sep=" ",
        usecols=[0, 1, 2, 3],
        names=["red", "green", "blue", "label"],
    )
    x = data.drop("label", axis=1)  # 8 columns of dataset
    y = data["label"]  # label
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=0, shuffle=True
    )
    res = []
    svclassifier = SVC(
        kernel="linear", gamma=2
    )  # gamma is Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    svclassifier.fit(x_train, y_train)

    y_pred = svclassifier.predict(f1)
    pred = []
    for i in range(0, len(y_pred)):
        pred.append(y_pred[i])
    f1["label"] = pred
    np.savetxt(r"pred_dataset.txt", f1.values, fmt="%d")




def image_to_dataset():
    """
    It is a function that makes a dataset which contains of images' RGB values that is labeld too.
    This dataset is a combination of images which contain fire and do not contain fire.
    The result of this function is saved in image_dataset.txt.
    """
    # call functions
    f1 = df_fire("fire.jfif")
    f2 = df_fire("fire_2.jpg")
    f3 = df_fire("fire_3.jpg")
    f4 = f1._append(f2, ignore_index=True)
    f5 = f4._append(f3, ignore_index=True)  # all of the pixels
    fn1 = df_nofire("demo_2.jpg")
    fn2 = df_nofire("river.jpg")
    fn4 = fn1._append(fn2, ignore_index=True)
    fn6 = f5._append(fn4, ignore_index=True)
    np.savetxt(r"image_dataset.txt", fn6.values, fmt="%d")


def df_fire(picture):
    """
    it is a function for getting pixels of image that contains the fire and labeling them
    """
    pixels_list = image_to_numerical_rgb(picture)
    label_list = []
    arr = np.array(pixels_list)  # convert rgb list to array
    for i in arr:
        label_list.append(1)
    dataframe = pd.DataFrame(
        pixels_list, columns=[0, 1, 2]
    )  # add the array to dataframe
    dataframe["label"] = label_list  # add label to dataframe
    return dataframe


def df_fire_d(picture):
    """
    It is a function that get the pixels of the image that should be used for prediction of it's label
    """
    pixels_list = image_to_numerical_rgb(picture)
    dataframe = pd.DataFrame(
        pixels_list, columns=[0, 1, 2]
    )  # add the array to dataframe
    return dataframe


def df_nofire(picture):
    """
    it is a function for getting pixels of image that does not contain the fire and labeling them
    """
    pixels_list = image_to_numerical_rgb(picture)
    label_list = []
    arr = np.array(pixels_list)  # convert rgb list to array
    for i in arr:
        label_list.append(2)
    dataframe = pd.DataFrame(
        pixels_list, columns=[0, 1, 2]
    )  # add the array to dataframe
    dataframe["label"] = label_list  # add label to dataframe
    return dataframe


def image_to_numerical_rgb(picture):
    pic = imageio.imread(picture)
    h = pic.shape[0]  # height of image
    w = pic.shape[1]  # width of image
    pixels_list = []
    if h < w:
        for i in range(0, h):
            for j in range(0, w):
                pixels_list.append(pic[i, j])  # add rgb of pixels to a list

    else:
        for i in range(0, h):
            for j in range(0, w):
                pixels_list.append(pic[i, j])
    return pixels_list


if __name__ == "__main__":
    main()
