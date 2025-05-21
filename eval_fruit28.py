# eval_fruit28.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix

# 1. 參數，請跟訓練時保持一致
MODEL_PATH = "fruit28_AUG_model.keras"
DATA_DIR   = "images"      # 訓練時放圖片的資料夾
IMG_SIZE   = 64
BATCH      = 128
VAL_SPLIT  = 0.2

# 2. 載入已訓練並儲存的模型（含 compile 設定）
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ 模型載入完成：", MODEL_PATH)

# 3. 重建 validation generator
datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=VAL_SPLIT
)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
print("✅ 驗證集生成器建立完成，共", val_gen.samples, "張圖片")

# 4. 在驗證集上評估
loss, acc = model.evaluate(val_gen, verbose=1)
print(f"\n🔍 驗證集結果 → Loss = {loss:.4f}, Accuracy = {acc:.4%}")

# --- 評估後取得 y_true, y_pred ---

# --- 取出 ground truth 與模型預測 ---
y_true = val_gen.classes                           # 直接拿到所有驗證集標籤 (整數 0–27)
preds   = model.predict(val_gen, verbose=0)        # shape=(n_samples, 28)
y_pred  = np.argmax(preds, axis=1)                 # 取每列最大機率的索引

# --- 建立 idx→label 的對照表 ---
class_indices = val_gen.class_indices              # dict: {'Apple_Healthy':0, ...}
idx2label     = {v:k for k,v in class_indices.items()}
labels        = [idx2label[i] for i in range(len(idx2label))]

# --- 列印各類別詳細報告 ---
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, target_names=labels))

# --- 列印混淆矩陣 ---
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
