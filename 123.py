import tensorflow as tf

# 1️⃣ 在支援 BF16 的環境載入
tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
model = tf.keras.models.load_model("fruit28_EffB0_bf16.keras")

# 2️⃣ 切回 float32
tf.keras.mixed_precision.set_global_policy("float32")
model.compile(optimizer="adam", loss="categorical_crossentropy")

# 3️⃣ 重新存檔
model.save("fruit28_EffB0_fp32.keras")
print("✅ 轉存完成：fruit28_EffB0_fp32.keras")
