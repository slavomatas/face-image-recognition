import os
from glob import glob

import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.initializers import TruncatedNormal
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Dense
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.datasets import load_files
from tqdm import tqdm


def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255
    return img_tensor


def load_images(img_paths):
    list_of_tensor_images = [load_image(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensor_images)


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


def create_model():
    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GlobalAveragePooling2D())

    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_initializer=init,
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu',
                    kernel_initializer=init,
                    bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(133, activation='softmax',
                    kernel_initializer=init,
                    bias_initializer='zeros'))

    model.summary()

    # Adam optimizer
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    #model.compile(loss='categorical_crossentropy',
    #              optimizer='rmsprop',
    #              metrics=['accuracy'])

    return model


def train_model():
    base_dir = '/home/slavo/Dev/face-image-recognition/dogImages/'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'valid')

    batch_size = 20

    # Augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    # this is a generator that will read pictures found in subfolders of 'data/train',
    # and indefinitely generate batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    # this is the augmentation configuration we will use for testing: only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    # ### Train the Model
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                                   verbose=1, save_best_only=True)

    model = create_model()

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=500,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[checkpointer],
        verbose=1,
        use_multiprocessing=1,
        workers=5)


def test_model():
    # Load train, test, and validation datasets
    train_files, train_targets = load_dataset('/home/slavo/Dev/face-image-recognition/dogImages/train')
    valid_files, valid_targets = load_dataset('/home/slavo/Dev/face-image-recognition/dogImages/valid')
    test_files, test_targets = load_dataset('/home/slavo/Dev/face-image-recognition/dogImages/test')

    # Load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("/home/slavo/Dev/face-image-recognition/dogImages/train/*/"))]

    # print statistics about the dataset
    print('There are %d total dog categories.' % len(dog_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.' % len(test_files))

    import random
    random.seed(8675309)

    # Load filenames in shuffled human dataset
    human_files = np.array(glob("/home/slavo/Dev/face-image-recognition/lfw/*/*"))
    random.shuffle(human_files)

    # Print statistics about the dataset
    print('There are %d total human images.' % len(human_files))

    # Create model
    model = create_model()

    # Load the model weights with the best validation Loss
    #model.load_weights('saved_models/weights.best.rmsprop.hdf5')
    model.load_weights('saved_models/weights.best.from_scratch.hdf5')

    # Evaluate Model
    print(model.evaluate(load_images(test_files), test_targets))

    # Get index of predicted dog breed for each image in test set
    model_predictions = [np.argmax(model.predict(load_image(test_file))) for test_file in test_files]

    # Report test accuracy
    test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(test_targets, axis=1)) / len(model_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

    # TODO: Test the performance of the dog_detector function
    # on the images in human_files_short and dog_files_short.

    dog_files_short = train_files[:100]
    dog_targets_short = train_targets[:100]
    human_files_short = human_files[:100]

    model_predictions = []
    for dog_file, dog_target in zip(dog_files_short, dog_targets_short):
        model_prediction = np.argmax(model.predict(load_image(dog_file)))
        model_predictions.append(model_prediction)
        #print("Dog prediction:", model_prediction, " Target:", np.argmax(dog_target))

    # Report test accuracy on dog files
    dog_test_accuracy = 100 * np.sum(np.array(model_predictions) == np.argmax(dog_targets_short, axis=1)) / len(model_predictions)
    print('Dog test accuracy: %.4f%%' % dog_test_accuracy)

    model_predictions = []
    for human_file in human_files_short:
        model_prediction = np.argmax(model.predict(load_image(human_file)))
        model_predictions.append(model_prediction)
        #print("Human prediction:", model_prediction, " Target:", 0)


    # Report test accuracy on human files
    human_test_accuracy = 100 * np.sum(np.array(model_predictions) == 0) / len(model_predictions)
    print('Human test accuracy: %.4f%%' % human_test_accuracy)

#train_model()
test_model()
