# Fruit28 Classifier

本專案使用 TensorFlow 訓練並推論 28 種水果的分類模型。

## 環境需求

- **作業系統**：Windows / Linux / macOS  
- **Python**：3.8 或以上  
- **記憶體**：≥8 GB  
- **GPU（選用）**：NVIDIA CUDA 支援顯卡，訓練時建議使用  
- **工具**：git、VS Code（或其他 IDE）

## 安裝步驟

1. **建立並啟用虛擬環境**  
   - Windows (cmd)：  
     ```bat
     python -m venv venv
     venv\Scripts\activate
     ```  
   - Linux/macOS (bash)：  
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

2. **安裝所需 Python 套件**  
   - 確保 `requirements.txt` 已在專案根目錄（若無請參考第 1 步）。  
   - 執行：
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

3. **驗證安裝**  
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
   python -c "import cv2; print('OpenCV:', cv2.__version__)"
