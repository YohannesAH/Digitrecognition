import numpy as np
import mnist
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

import keras.models as models
import matplotlib.pyplot as plt

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = Sequential([
  Conv2D(8, 3, input_shape=(28, 28, 1), use_bias=False),
  MaxPooling2D(pool_size=2),
  Flatten(),
  Dense(10, activation='softmax'),
])

model.compile(SGD(lr=.005), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2
)

model.fit_generator(
    datagen.flow(train_images, to_categorical(train_labels), batch_size=32),
    steps_per_epoch=len(train_images) / 32,
    epochs=10,
    validation_data=(test_images, to_categorical(test_labels))
)

# Save the model
model.save('mymodel')

#load the model
#model = models.load_model('mymodel')

'''
Epoch 1
46s 760us/step - loss: 0.2433 - acc: 0.9276 - val_loss: 0.1176 - val_acc: 0.9634
Epoch 2
46s 771us/step - loss: 0.1184 - acc: 0.9648 - val_loss: 0.0936 - val_acc: 0.9721
Epoch 3
48s 797us/step - loss: 0.0930 - acc: 0.9721 - val_loss: 0.0778 - val_acc: 0.9744
'''
# Test my image
print('\n--- Testing the CNN with my image ---')

# Read the image using opencv to make my life easy
img = cv2.imread('images_mine/3.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28,28)) # Resize - important! 
img = cv2.bitwise_not(img)
img = (img / 255) - 0.5  # The other version does this in the convolution forward() function

# Try to use the image.load_img() function from keras utils as well
#img = image.load_img('images_mine/0.bmp', color_mode = "grayscale", target_size=(28, 28))
#to_grayscale = keras_cv.layers.preprocessing.Grayscale()
#img = to_grayscale(img)
#...

img_tensor = np.expand_dims(img, axis=0)

prediction = model.predict(img_tensor)
print(prediction)
classes=np.argmax(prediction,axis=1)
print(classes)

plt.imshow(img_tensor[0], cmap=plt.get_cmap('gray_r'))
plt.show()


# keep going with the test images...

test_images = mnist.test_images()[100:]
test_images = (test_images / 255) - 0.5

for im, label in zip(test_images, test_labels):
    img_tensor = np.expand_dims(im, axis=0)

    prediction = model.predict(img_tensor)
    print(prediction)

    classes=np.argmax(prediction,axis=1)
    print(classes)

    plt.imshow(img_tensor[0], cmap=plt.get_cmap('gray_r'))
    plt.show()

