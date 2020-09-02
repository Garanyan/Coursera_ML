import os
import zipfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

# https://www.kaggle.com/c/dogs-vs-cats/overview

DESTINATION_DIR = "dogs-vs-cats"
ZIPFILE = "dogs-vs-cats.zip"

if not os.path.exists(DESTINATION_DIR):
    zip_ref = zipfile.ZipFile(ZIPFILE, 'r')
    zip_ref.extractall(DESTINATION_DIR)
    zip_ref.close()

train_dir = os.path.join(DESTINATION_DIR, 'train')

if not os.path.exists(train_dir):
    zip_ref = zipfile.ZipFile(DESTINATION_DIR + "/train.zip")
    zip_ref.extractall(train_dir)
    zip_ref.close()

test_dir = os.path.join(DESTINATION_DIR, 'test')
if not os.path.exists(test_dir):
    zip_ref = zipfile.ZipFile(os.path.join(DESTINATION_DIR, "test1.zip"))
    zip_ref.extractall(test_dir)
    zip_ref.close()

cats_dir = os.path.join(train_dir, 'cats')
dogs_dir = os.path.join(train_dir, 'dogs')

cats_fnames = os.listdir(cats_dir)
dogs_fnames = os.listdir(dogs_dir)

print(f"Cats {len(cats_fnames)} imgs.", cats_fnames[:10])
print(f"Dogs {len(dogs_fnames)} imgs.", dogs_fnames[:10])


def show_imgs():
    nrows = 4
    ncols = 4
    pic_index = 0  # Index for iterating over images
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8

    next_cat_pix = [os.path.join(cats_dir, fname)
                    for fname in cats_fnames[pic_index - 8:pic_index]
                    ]

    next_dog_pix = [os.path.join(dogs_dir, fname)
                    for fname in dogs_fnames[pic_index - 8:pic_index]
                    ]

    for i, img_path in enumerate(next_cat_pix + next_dog_pix):
        # Set up subplot; subplot indices start at 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
datagen = ImageDataGenerator(rescale=1.0 / 255., validation_split=0.3)
train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(150, 150),
                                              batch_size=50,
                                              class_mode='binary',
                                              subset='training')

val_generator = datagen.flow_from_directory(train_dir,
                                            target_size=(150, 150),
                                            batch_size=50,
                                            class_mode='binary',
                                            subset='validation')

history = model.fit_generator(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=120,
    epochs=15,
    validation_steps=50,
    verbose=2
)

model.save_weights('checkpoints/checkpoint_' + str(round(history.history['val_accuracy'][-1], 2)))


def plot_acc(history):
    # -----------------------------------------------------------
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')


# plot_acc(history)

test_files = os.listdir(test_dir)
test_files.sort()
print(f"test files {len(test_files)}", test_files[0:10])
with open("ans.csv", "w") as f:
    for file in test_files:
        img = image.load_img(os.path.join(test_dir, file), target_size=(150, 150, 3))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images)

        if classes[0] > 0:
            f.write(f"{file},{1}\n")
        else:
            f.write(f"{file},{0}\n")
