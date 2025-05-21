
#fruit28_ImageNet_tuned2.py


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import numpy as np, os, datetime

# ───────── 1. 全域設定 ──────────
IMG_SIZE   = 64
BATCH      = 64                         # <<< 修改：較小批次
EPOCHS_HEAD = 5                         # <<< 修改：凍結 5 epoch
EPOCHS_FT   = 25                        # 微調 25 epoch
TRAIN_DIR = r"C:\vegtable\fruit28_cls\images\train"
VAL_DIR   = r"C:\vegtable\fruit28_cls\images\val"

# ───────── 2. 資料增強 ──────────
train_gen = ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10
).flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode="categorical", shuffle=True)

val_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH, class_mode="categorical", shuffle=False)

NUM_CLASSES = train_gen.num_classes
print("總類別數量 =", NUM_CLASSES)

# ───────── 3. 計算 class_weight (0.8–2.0) ──────────
cls_idx    = np.array(list(train_gen.class_indices.values()))
cls_labels = train_gen.classes
w_raw      = compute_class_weight('balanced', classes=cls_idx, y=cls_labels)
w_clip     = np.clip(w_raw, 0.8, 2.0)               # <<< 修改：限幅
class_weight = dict(zip(cls_idx, w_clip))
print("class_weight:", class_weight)

# ───────── 4. 建 EfficientNet-B0 (凍結) ──────────
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
model = models.Model(inputs, outputs)

opt_head = tf.keras.optimizers.Adam(5e-4)           # <<< 修改：LR 5e-4
model.compile(opt_head, loss="categorical_crossentropy", metrics=["accuracy"])
print("初始 LR =", opt_head.learning_rate.numpy())
model.summary()

# ───────── 5. Callbacks ──────────
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_cb = TensorBoard(logdir, histogram_freq=1)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=1,       # <<< 修改：patience 1
    min_delta=1e-4, min_lr=1e-6)

early_stop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# ───────── 6. 訓練分類頭 (凍結 5 epoch) ──────────
history_head = model.fit(
    train_gen, epochs=EPOCHS_HEAD, validation_data=val_gen,
    class_weight=class_weight, callbacks=[reduce_lr, early_stop, tb_cb])

# ───────── 7. 微調：解凍最後 20 層 ──────────
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

history_ft = model.fit(
    train_gen, epochs=EPOCHS_FT, validation_data=val_gen,
    class_weight=class_weight, callbacks=[reduce_lr, early_stop, tb_cb])

# ───────── 8. 儲存 ──────────
model.save("fruit28_EffB0_ft32.keras")
print("✓ 模型已儲存")
