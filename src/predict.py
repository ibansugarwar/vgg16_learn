import os, glob
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential, Model, load_model
from common import get_classes

# 判別対象名（フォルダ名）の取得
classes = get_classes()

num_classes = len(classes)
image_size = 224

CURRENT_DIR = os.path.abspath('.')

model = load_model(f'{CURRENT_DIR}/h5/vgg16_classification.h5')
    
# 引数から画像ファイルを参照して読み込む
files = glob.glob(f"{CURRENT_DIR}/target/*.jpg")
for file in files:
    image = Image.open(file)
    image = image.convert("RGB")
    
    # 余白追加処理
    width, height = image.size
    background_color = (0,0,0)
    if width > height:
        padding_image = Image.new(image.mode, (width, width), background_color)
        padding_image.paste(image, (0, (width - height) // 2))
        image = padding_image
    if width < height:
        padding_image = Image.new(image.mode, (height, height), background_color)
        padding_image.paste(image, ((height - width) // 2, 0))
        image = padding_image
        
    image_224 = image.resize((image_size,image_size))
    data = np.asarray(image_224) * 255.0
    X = []
    X.append(data)
    X_np = np.array(X)

    result = model.predict([X_np])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)

    # 判定結果の出力
    if predicted >= num_classes:
        print("判別対象外", percentage, file)
    else:
        print(classes[predicted], percentage, file)
