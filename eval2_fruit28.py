import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# 1️⃣ 基本參數
MODEL_PATH = "fruit28_gpu_aug.keras"
VAL_DIR    = r"C:\vegtable\fruit28_cls\images\val"   # ← 直接指向 val
IMG_SIZE   = 128
BATCH      = 128

# 2️⃣ 載入模型
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 模型載入完成：", MODEL_PATH)

# 3️⃣ 只做 rescale 的驗證集 generator
val_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    shuffle=False
)
print(f"✅ 驗證集生成器建立完成，共 {val_gen.samples} 張圖片")

# 4️⃣ 評估
loss, acc = model.evaluate(val_gen, verbose=1)
print(f"\n🔍 驗證集結果 → Loss = {loss:.4f}, Accuracy = {acc:.4%}")

# 5️⃣ 取得 y_true / y_pred，列印報告與混淆矩陣
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

labels = list(val_gen.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

