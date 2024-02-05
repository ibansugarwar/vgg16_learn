import os, yaml

CURRENT_DIR = os.path.abspath('.')

def get_classes():
    # 判別対象画像のフォルダ名取得
    train_target_info = f'{CURRENT_DIR}/yml/train_target_info.yml'
    with open(train_target_info, 'r') as yml:
        train_target_info = yaml.safe_load(yml)
    classes = [item['target_folder_name'] for item in train_target_info]
    
    return classes