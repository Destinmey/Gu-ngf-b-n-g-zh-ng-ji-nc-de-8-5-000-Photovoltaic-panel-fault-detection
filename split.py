import os
import random
import shutil

# 设置随机数种子
random.seed(123)

# 定义文件夹路径
root_dir = 'train'
image_dir = os.path.join(root_dir, 'images')
label_dir = os.path.join(root_dir, 'labels')
output_dir = 'splitdata'

# 定义训练集、验证集和测试集比例
train_ratio = 0.8
valid_ratio = 0.2

# 获取所有图像文件和标签文件的文件名（不包括文件扩展名）
image_filenames = [os.path.splitext(f)[0] for f in os.listdir(image_dir)]
label_filenames = [os.path.splitext(f)[0] for f in os.listdir(label_dir)]

# 随机打乱文件名列表
random.shuffle(image_filenames)

# 计算训练集、验证集和测试集的数量
total_count = len(image_filenames)
train_count = int(total_count * train_ratio)
valid_count = int(total_count * valid_ratio)
test_count = total_count - train_count - valid_count

# 定义输出文件夹路径
train_image_dir = os.path.join(output_dir, 'images', 'train')
train_label_dir = os.path.join(output_dir, 'labels', 'train')
valid_image_dir = os.path.join(output_dir, 'images', 'val')
valid_label_dir = os.path.join(output_dir, 'labels', 'val')

# 创建输出文件夹
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(valid_image_dir, exist_ok=True)
os.makedirs(valid_label_dir, exist_ok=True)

# 将图像和标签文件划分到不同的数据集中
for i, filename in enumerate(image_filenames):
    if i < train_count:
        output_image_dir = train_image_dir
        output_label_dir = train_label_dir
    elif i < train_count + valid_count:
        output_image_dir = valid_image_dir
        output_label_dir = valid_label_dir


    # 复制图像文件
    src_image_path = os.path.join(image_dir, filename + '.jpg')
    dst_image_path = os.path.join(output_image_dir, filename + '.jpg')
    shutil.copy(src_image_path, dst_image_path)

    # 复制标签文件
    src_label_path = os.path.join(label_dir, filename + '.txt')
    dst_label_path = os.path.join(output_label_dir, filename + '.txt')
    shutil.copy(src_label_path, dst_label_path)
