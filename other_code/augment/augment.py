from keras.preprocessing.image import ImageDataGenerator

path = "cropped_images"

datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

BATCH_SIZE = 32
gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

gen = datagen.flow_from_directory(path, target_size=(140,140), color_mode="rgba", batch_size=32)

