import os, glob
import numpy as np

from PIL import Image
from sklearn import model_selection
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model           
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import np_utils

from common import get_classes

IMAGE_SIZE = 224
CURRENT_DIR = os.path.abspath('.')

# 判別対象画像のフォルダ名取得
classes = get_classes()

# 画像の読み込みとNumPy配列への変換
X = []
Y = []
for index, classlabel in enumerate(classes):
    imgs_dir = f'{CURRENT_DIR}/img/{classlabel}'
    files = glob.glob(imgs_dir + '/*.jpg')
    for i, file in enumerate(files):
        # 進捗表示用
        print(f'{i + 1}/{len(files)}', file)
        
        # 画像ファイル読み込み
        image = Image.open(file)
        # 色情報を揃える
        image = image.convert('RGB')
        
        # 正方形になるよう上下または左右に余白（黒色）を追加
        width, height = image.size
        background_color = (0,0,0)
        if width > height:
            result = Image.new(image.mode, (width, width), background_color)
            result.paste(image, (0, (width - height) // 2))
            image = result
        if width < height:
            result = Image.new(image.mode, (height, height), background_color)
            result.paste(image, ((height - width) // 2, 0))
            image = result

        # 全ての画像のサイズを同じ大きさにする
        image = image.resize((IMAGE_SIZE,IMAGE_SIZE))
        array_data = np.asarray(image)
        
        X.append(array_data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

# 学習用、テスト用に分割
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)

# One-hotベクトル化
num_classes = len(classes)
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# 正規化
X_train = X_train.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

# モデルの定義
model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
model.summary()
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_classes, activation='softmax'))

model = Model(inputs=model.input, outputs=top_model(model.output))

model.summary()
for layer in model.layers[:15]:
    layer.trainable = False

early_stopping = EarlyStopping(monitor='accuracy', patience=3, restore_best_weights=True)

opt = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=20, callbacks=[early_stopping])

h5_file = f'{CURRENT_DIR}/h5/vgg16_classification.h5'
model.save(h5_file)