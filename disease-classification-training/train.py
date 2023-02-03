import os
import argparse
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dropout,Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Lambda, Input, AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import shutil
import cv2
import os
import kaggle

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""disease_classification for xray image and categorical data""")

    parser.add_argument(
        "--initial learning rate",
        action="store",
        dest="lr",
        required=True,
        help="""--- initial learning rate ---""",
    )
    parser.add_argument(
        "--epochs",
        action="store",
        dest="epochs",
        required=True,
        help="""--- number of epochs ---""",
    )
    parser.add_argument(
        "--model_name",
        action="store",
        dest="model_name",
        required=True,
        help="""--- model_name ---""",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        dest="batch_size",
        required=False,
        default="32",
        help="""--- batch size for training""",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save the dataset file to ---""",
    )

    return parser.parse_args()


# custom exceptions
class EmptyDataError(Exception):
    """Raise if there is no data"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "EmptyDataError: The query resulted in empty data, please check if the bucket is empty"


class IncorrectDirectoryError(Exception):
    """Raise if directory is invalid"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "IncorrectDirectoryError: Please ensure the input directory is valid!"

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

samples = 25

# optional if S3 is not used
'''
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

### kaggle.api.dataset_download_files('bachrr/covid-chest-xray', path='/cnvrg', unzip=True)
kaggle.api.dataset_download_files('bachrr/covid-chest-xray', path='./input', unzip=True)
kaggle.api.dataset_download_files('paultimothymooney/chest-xray-pneumonia', path='./input', unzip=True)
'''

def train_model(lr, epochs, batch, model_name, output_dir):
    if model_name == "vgg16":
        from tf.keras.applications.vgg16 import VGG16
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    if model_name == "resnet50":
        from tf.keras.applications.resnet50 import ResNet50
        baseModel = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    if model_name == "resnet101":
        from tf.keras.applications.resnet import ResNet101
        baseModel = ResNet101(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    dataset_path = './dataset'
    current_directory = os.getcwd()

    final_directory = os.path.join(current_directory, r'dataset/covid')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    
    covid_dataset_path = '../input/covid-chest-xray'

    final_directory = os.path.join(current_directory, r'dataset/normal')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    covid_dataset_path = '.'

    # construct the path to the metadata CSV file and load it
    csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
    df = pd.read_csv(csvPath)

    # loop over the rows of the COVID-19 data frame
    for (i, row) in df.iterrows():
        # if (1) the current case is not COVID-19 or (2) this is not
        # a 'PA' view, then ignore the row
        if row["finding"] != "COVID-19" or row["view"] != "PA":
            continue

        # build the path to the input image file
        imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

        # if the input image file does not exist (there are some errors in
        # the COVID-19 metadeta file), ignore the row
        if not os.path.exists(imagePath):
            continue

        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = row["filename"].split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

        # copy the image
        shutil.copy2(imagePath, outputPath)



    pneumonia_dataset_path ='../input/chest-xray-pneumonia/chest_xray'
    basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
    imagePaths = list(paths.list_images(basePath))

    # randomly sample the image paths
    random.seed(42)
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:samples]

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = imagePath.split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

        # copy the image
        shutil.copy2(imagePath, outputPath)


    ### pre processing ####

    # construct the path to the metadata CSV file and load it
    csvPath = os.path.sep.join([covid_dataset_path, "metadata.csv"])
    df = pd.read_csv(csvPath)

    # loop over the rows of the COVID-19 data frame
    for (i, row) in df.iterrows():
        # if (1) the current case is not COVID-19 or (2) this is not
        # a 'PA' view, then ignore the row
        if row["finding"] != "COVID-19" or row["view"] != "PA":
            continue

        # build the path to the input image file
        imagePath = os.path.sep.join([covid_dataset_path, "images", row["filename"]])

        # if the input image file does not exist (there are some errors in
        # the COVID-19 metadeta file), ignore the row
        if not os.path.exists(imagePath):
            continue

        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = row["filename"].split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f"{dataset_path}/covid", filename])

        # copy the image
        shutil.copy2(imagePath, outputPath)

    pneumonia_dataset_path ='../input/chest-xray-pneumonia/chest_xray'

    basePath = os.path.sep.join([pneumonia_dataset_path, "train", "NORMAL"])
    imagePaths = list(paths.list_images(basePath))

    # randomly sample the image paths
    random.seed(42)
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:samples]

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the filename from the image path and then construct the
        # path to the copied image file
        filename = imagePath.split(os.path.sep)[-1]
        outputPath = os.path.sep.join([f"{dataset_path}/normal", filename])

        # copy the image
        shutil.copy2(imagePath, outputPath)

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_path))
    data = []
    labels = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 1]
    data = np.array(data) / 255.0
    labels = np.array(labels)

    #Modeling
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")

 
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.6)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=lr, decay=lr / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the head of the network
    print("[INFO] training head...")
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=batch),
        steps_per_epoch=len(trainX) // batch,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch,
        epochs=epochs)

    # save model
    model.save(output_dir + '/' + model_name + '.h5')

    # evaluation
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=batch)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
    # convert classification report to dataframe
    report = classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_, output_dict=True)
    classification_df = pd.DataFrame(report).transpose()
    
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))
    # convert acc, sensitivity, specificity to dataframe
    confusion_df = pd.DataFrame([[acc, sensitivity, specificity]], columns=['acc', 'sensitivity', 'specificity']) 

    return classification_df, confusion_df


def main():
    args = parse_parameters()
    if str(args.cnvrg_dataset).lower() != "none":
        if not os.path.exists(args.local_dir):
            raise IncorrectDirectoryError()
    # return pandas dataframe from custom query
    disease_classification = disease_classification(url=args.url, token=args.token, org=args.org)
    classification_df, confusion_df = disease_classification.train_model(
        args.lr,
        args.epoch,
        args.batch_size,
        args.model_name,
        args.output_dir
    )
    classification_df.to_csv(args.output_dir + '/classification.csv', index=False)
    confusion_df.to_csv(args.output_dir + '/confusion_df.csv', index=False)


if __name__ == "__main__":
    main()