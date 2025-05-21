import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
#參數

DATA_DIR = "images"
IMG_SIZE = 64
BATCH = 128
EPOCHS = 30
VAL_SPLIT = 0.2



#資料增強

datagen = ImageDataGenerator(
    rescale = 1/255.,
    validation_split = VAL_SPLIT,
    horizontal_flip = True,
    zoom_range = 0.1)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = BATCH,
    class_mode = "categorical",
    subset = "training")

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size = (IMG_SIZE,IMG_SIZE),
    batch_size = BATCH,
    class_mode = "categorical",
    subset = "validation")

NUM_CLASSES = train_gen.num_classes
print("總類別數量=",NUM_CLASSES)

#建立模型

model=models.Sequential([
    layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)),
    layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

logdir = os.path.join("logs",
         datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

#設定cllback早停,自動調整學習率

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # 觀察驗證損失
    factor=0.5,           # LR ×0.5
    patience=4,           # 4 連敗就降
    min_delta=1e-4,
    min_lr=1e-6
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

tb_cb = tf.keras.callbacks.TensorBoard(logdir,histogram_freq=1)

history = model.fit(
    train_gen,
    validation_data = val_gen,
    epochs = EPOCHS,
    callbacks = [tb_cb,reduce_lr,early_stop])

#儲存

model.save("fruit28_AUG_model.keras")
print("模型以儲存")
