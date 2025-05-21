# split_aug_original.py
# fruit28_split_train_val.py
from pathlib import Path
import shutil, random, re, os

# ───── 1. 基本設定 ─────
SOURCE_DIR  = Path(r"C:/vegtable/fruit28_cls/images_all")   # 28 類全集（原+aug）
DEST_ROOT   = Path(r"C:/vegtable/fruit28_cls/images")       # 輸出根
TRAIN_DIR   = DEST_ROOT / "train"
VAL_DIR     = DEST_ROOT / "val"

# 原圖→val 的比例 (0–1)。若想改「固定 N 張」，把 VAL_RATIO 改 None，VAL_LIMIT 設 N。
VAL_RATIO = 0.20
VAL_LIMIT = None         # ＝None 時以比例抽樣

AUG_PATTERN = re.compile(r"_aug\d+\.", re.IGNORECASE)   # 檔名含 _aug0. / _aug1. …

random.seed(42)          # 使抽樣可重現

# ───── 2. 先清空目的地資料夾 ─────
if DEST_ROOT.exists():
    shutil.rmtree(DEST_ROOT)
TRAIN_DIR.mkdir(parents=True)
VAL_DIR.mkdir(parents=True)

# ───── 3. 逐類切分 ─────
ext_list = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

for cls_dir in SOURCE_DIR.iterdir():
    if not cls_dir.is_dir():
        continue

    cls_name = cls_dir.name
    (TRAIN_DIR/cls_name).mkdir()
    (VAL_DIR  /cls_name).mkdir()

    # --- 收集檔案 ---
    files = []
    for ext in ext_list:
        files.extend(cls_dir.glob(ext))

    originals = [f for f in files if not AUG_PATTERN.search(f.name)]
    augments  = [f for f in files if     AUG_PATTERN.search(f.name)]

    MAX_AUG = 900                       # 想要留多少張增強圖就填多少
    if cls_name == "Apple__Healthy" and len(augments) > MAX_AUG:
        random.shuffle(augments)         # 隨機挑
        augments = augments[:MAX_AUG]

    # --- 決定 val / train 原圖 ---
    random.shuffle(originals)

    if VAL_LIMIT is not None:                 # 固定張數模式
        val_files   = originals[:VAL_LIMIT]
        train_files = originals[VAL_LIMIT:]
    else:                                     # 比例抽樣模式
        k = max(1, int(len(originals) * VAL_RATIO))
        val_files   = originals[:k]
        train_files = originals[k:]

    # --- 複製檔案 ---
    for f in val_files:
        shutil.copy2(f, VAL_DIR/cls_name/f.name)
    for f in train_files + augments:          # aug 全丟 train
        shutil.copy2(f, TRAIN_DIR/cls_name/f.name)

    print(f"{cls_name:25s}  "
          f"val:{len(val_files):4d}  "
          f"train_orig:{len(train_files):4d}  "
          f"train_aug:{len(augments):4d}")

# ───── 4. 總結 ─────
def count_imgs(root):
    return sum(len(list((root/c).iterdir())) for c in root.iterdir())

print("\n✅ 切分完成")
print(f"  - train 影像總數：{count_imgs(TRAIN_DIR)}")
print(f"  - val   影像總數：{count_imgs(VAL_DIR)}")
