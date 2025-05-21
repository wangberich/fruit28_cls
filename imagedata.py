from pathlib import Path
import cv2,albumentations as A, numpy as np
from tqdm import tqdm
#資料夾結構
ROOTS = Path('C:/vegtable/fruit28_cls/images')
TARGET_CLASSES = [
    'Apple__Healthy',
    'Bellpepper__Rotten',
    'Pomegranate__Rotten',
    'Tomato__Healthy',
    'Tomato__Rotten'    
]
#定義增強項目

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=0.2,p=0.5),
    A.GaussNoise(var_limit=(10.0,50.0),p=0.3),
    A.RandomShadow(p=0.3),
],additional_targets={'image':'image'})

MULTIPLY = 3 #設定每張圖片要在生成幾張

for cls in TARGET_CLASSES:
    cls_dir = ROOTS / cls
    imgs = list(cls_dir.glob('*'))

    for img_path in tqdm(imgs,desc = cls):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        for i in range(MULTIPLY):
            augmented = augment(image=image)['image']
            out_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            cv2.imwrite(str(cls_dir/out_name),
                        cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

#A.Compose([...])：把多個增強操作組成「管線」
#子增強	功能	常用參數重點
#HorizontalFlip(p=0.5)	隨機左右翻轉	p＝套用機率 0~1。
#RandomRotate90(p=0.5)	隨機把圖旋轉 0°/90°/180°/270°	
#RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)	隨機調亮度與對比	limit 設 ±0.2 → 亮度或對比最多 ±20%。
#GaussNoise(var_limit=(10,50), p=0.3)	加高斯雜訊	var_limit 控制雜訊強度。
#RandomShadow(p=0.3)	在隨機位置畫出陰影	無其他必填參數。
#additional_targets={'image':'image'}
#這裡只是佔位用，若有第二張影像（如 mask）須同步增強才會派上用場；單純分類任務寫不寫皆可。

###常見增強函數速查

#| 函數                          | 效果                 | 主要參數                                     |
#| --------------------------- | ------------------ | ---------------------------------------- |
#| `RandomCrop` / `CenterCrop` | 隨機／置中裁剪              | `height, width`                          |
#| `RandomResizedCrop`         | 裁剪後再縮放回原大小         | `scale, ratio`                           |
#| `ShiftScaleRotate`          | 平移、縮放、旋轉一次搞定     | `shift_limit, scale_limit, rotate_limit` |
#| `CLAHE`                     | 直方圖均衡化                | `clip_limit, tile_grid_size`             |
#| `Blur` / `GaussianBlur`     | 模糊                       | `blur_limit`                             |
#| `Cutout` / `CoarseDropout`  | 隨機遮擋區域                | `max_holes, max_height, max_width`       |
#| `ColorJitter`               | 亮度 / 對比 / 色調 / 飽和度 | `brightness, contrast, saturation, hue`  |


#_____________

#三重 for 迴圈的「逐層」工作流程
#（以 外 → 中 → 內 三層說明，搭配變數意義與實際動作）

#迴圈層級	迴圈變數	              每次迴圈在做什麼	                                                       主要副作用

#外層	cls ← TARGET_CLASSES	     1. 依序取得五個目標類別名稱（例如 'Apple__Healthy'）。               把後續所有動作鎖定在這個類別的資料夾內
#                                    2. 拼出該類別資料夾路徑 cls_dir = ROOTS / cls。	
#中層	img_path ← list(cls_dir.glob('*'))	1. 列舉此資料夾內「每一張原始圖片」路徑。               把單張影像載入為 Numpy 陣列 image
#                                           2. 讀檔 → BGR→RGB (cv2.imread ➜ cv2.cvtColor)。	
#內層	i ← range(MULTIPLY)（0, 1, 2）	1. 隨機增強：augmented = augment(image=image)['image']。  為該原圖生成 MULTIPLY 張「變形後的新圖片」，並寫檔
#                                       2. 產生新檔名：out_name = <原檔名>_aug<i>.jpg。
#                                       3. RGB→BGR，再 cv2.imwrite() 存回 同一類別資料夾。	


#| 行號                                    | 函式 / 語法                                
#| -------------------------------------- | -------------------------------------- 
#| ① `Path.glob('*')`                     | 回傳資料夾內所有檔案 (`Path` 物件清單)。   
#| ② `tqdm(imgs, desc=cls)`               | 把第二層迴圈包進進度條；`desc` 顯示目前類別名。          
#| `cv2.imread()`                         | 讀檔，得到 **BGR** 陣列。                      
#| `cv2.cvtColor(..., cv2.COLOR_BGR2RGB)` | OpenCV → Albumentations 需 **RGB**，故先轉。
#| ③ `augment(image=image)`               | 執行隨機增強，傳回字典 `{ 'image': <增強後影像> }`。  
#| `img_path.stem`                        | 檔名不含副檔名（`apple1`）。                  
#| `img_path.suffix`                      | 副檔名（`.jpg`）。                           
#| `cv2.imwrite()`                        | 把陣列寫回磁碟。寫檔前再把 RGB 轉回 BGR。            

#cls_dir = ROOTS / cls 必要嗎？	       必要，用來定位當前類別的資料夾。
##glob('*') 做什麼？	               把資料夾裡 所有檔案路徑列成清單。
#cv2.imread(str(img_path)) 為何轉字串？	OpenCV 舊版不認 Path，保險做法。功能是「讀圖」。
#augment(image=image)['image'] 是？	   把圖經過 隨機影像增強 後取出的新圖；去掉就不做增強。
#f-string 裡的 {i}？	                第 3 層迴圈的索引，用來給增強版本編號。
#imwrite() 會覆蓋嗎？	                不會，因檔名加了 _aug<i>；除非原資料夾已有叫同名的檔。