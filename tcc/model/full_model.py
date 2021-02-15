# Model imported from Google Colab
# Link: https://colab.research.google.com/drive/1YkG4ohQim2LHSaIamlPUNLz22EJF5XCf

from sklearn.metrics import plot_confusion_matrix, classification_report
import kerastuner as kt
import os
import tensorflow as tf
import numpy as np
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

save = False
optimize = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'save':
        save = True
    elif sys.argv[1] == 'optimize':
        optimize = True

# Global variables
EPOCHS = 20
IMAGE_SIZE_PROPORTION = 2 # A number between 2 and 5
EVENTS = ['penalty', 'freekick', 'celebration', 'corner', 'none']
NUM_OF_CLASSES = len(EVENTS)

# Image folders
train_penalty_dir = 'D:/TCC/images/train/penalty'
train_corner_dir = 'D:/TCC/images/train/corner'
train_fk_dir = 'D:/TCC/images/train/freekick'
train_none_dir = 'D:/TCC/images/train/none'
train_celebration_dir = 'D:/TCC/images/train/celebration'
valid_penalty_dir = 'D:/TCC/images/valid/penalty'
valid_corner_dir = 'D:/TCC/images/valid/corner'
valid_fk_dir = 'D:/TCC/images/valid/freekick'
valid_none_dir = 'D:/TCC/images/valid/none'
valid_celebration_dir = 'D:/TCC/images/valid/celebration'

train_penalty_names = os.listdir(train_penalty_dir)
train_corner_names = os.listdir(train_corner_dir)
train_fk_names = os.listdir(train_fk_dir)
train_none_names = os.listdir(train_none_dir)
train_celebration_names = os.listdir(train_celebration_dir)
valid_penalty_names = os.listdir(valid_penalty_dir)
valid_corner_names = os.listdir(valid_corner_dir)
valid_fk_names = os.listdir(valid_fk_dir)
valid_none_names = os.listdir(valid_none_dir)
valid_celebration_names = os.listdir(valid_celebration_dir)

print('Total Penalty Kick train images: {}'.format(len(train_penalty_names)))
print('Total Corner Kick train images: {}'.format(len(train_corner_names)))
print('Total Free Kick train images: {}'.format(len(train_fk_names)))
print('Total Non-Event train images: {}'.format(len(train_none_names)))
print('Total Celebration train images: {}'.format(len(train_celebration_names)))
print('Total Penalty Kick valid images: {}'.format(len(valid_penalty_names)))
print('Total Corner Kick valid images: {}'.format(len(valid_corner_names)))
print('Total Free Kick valid images: {}'.format(len(valid_fk_names)))
print('Total Non-Event valid images: {}'.format(len(valid_none_names)))
print('Total Celebration valid images: {}'.format(len(valid_celebration_names)))

# Data generators creation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

# Processed images has size of approximately 1367x761
# Dividing by 5 = 273x152
# Dividing by 4 = 341x190
# Dividing by 3 = 455x253
# Dividing by 2 = 688x380

resized_image = (int(1367/IMAGE_SIZE_PROPORTION), int(761/IMAGE_SIZE_PROPORTION))
resized_image_with_shape = (int(1367/IMAGE_SIZE_PROPORTION), int(761/IMAGE_SIZE_PROPORTION), 3)

train_generator = train_datagen.flow_from_directory(
        'D:/TCC/images/train/',  # This is the source directory for training images
        classes=EVENTS,
        target_size=resized_image,  # Todas as imagens serão redimensionadas para o tamanho definido por resized_image
        batch_size=16,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        'D:/TCC/images/valid/',  # This is the source directory for validation images
        classes=EVENTS,
        target_size=resized_image,  # Todas as imagens serão redimensionadas para o tamanho definido por resized_image
        batch_size=8,
        class_mode='categorical',
        shuffle=False)

if optimize:
    hypermodel = kt.applications.HyperResNet(input_shape=resized_image_with_shape, classes=NUM_OF_CLASSES)

    tuner = kt.Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=30,
        directory='keras_tuner_trials',
        project_name='tcc2')
    
    tuner.search(train_generator,
             validation_data=validation_generator,
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])
    
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

else:
    # Model creation
    leakyrelu_alpha = 0.5
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='linear', padding='same', input_shape=resized_image_with_shape),
                                        tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha),
                                        tf.keras.layers.MaxPooling2D((2, 2),padding='same'),
                                        tf.keras.layers.Conv2D(64, (3, 3), activation='linear',padding='same'),
                                        tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha),
                                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation='linear'),
                                        tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha),
                                        tf.keras.layers.Dense(NUM_OF_CLASSES, activation='softmax')])

    model.summary()

if optimize:
  model = tuner.hypermodel.build(best_hps)
else:
  model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='rmsprop',metrics=['accuracy'])

model.fit(train_generator,
        steps_per_epoch=(train_generator.n//train_generator.batch_size),
        epochs=EPOCHS,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=(validation_generator.n//validation_generator.batch_size))
model.evaluate(validation_generator)

STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator, steps=STEP_SIZE_TEST, verbose=1)
results = np.argmax(preds, axis=1)

print(plot_confusion_matrix(model, validation_generator.classes, results))
print('Classification Report')
print(classification_report(validation_generator.classes, results, target_names=EVENTS))

# serialize model to JSON
if save:
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")