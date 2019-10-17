Well done. The next step you need to create a generator which will read your current images and yield the augmented images. Keras has that implemented for you.

from keras.preprocessing.image import ImageDataGenerator

You will use ImageDataGenerator to create an instance datagen. The transformation will be specified at this step. Below is an example. 
```
datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
```          
Then you specify the training data and create the generator. The batch_size specifies how many samples are generated in one yield.
```
BATCH_SIZE = 32
gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)
```
Since the data is now coming from the generator gen (not numpy arrays), you need to use the model.fit_generator() for training, instead of the model.fit(). 
```
EPOCHS = 100
model.fit_generator(gen,
                    steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                    epochs=EPOCHS)
```
Note that the generator is created to run infinitely, so you need to specify how many batches equals to one epoch (steps_per_epoch). This is because the model only knows how many steps are performed but doesn't know when is one epoch. 

For more information, please first check the [documentation](https://keras.io/preprocessing/image/). Or ask me.
