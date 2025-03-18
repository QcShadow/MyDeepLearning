import os
from PIL import Image

def resize_images_in_folder(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        # 检查文件是否为图像文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 打开图像
                image = Image.open(file_path)
                # 调整图像大小，使用抗锯齿算法以保留更多信息
                resized_image = image.resize((300, 300), Image.LANCZOS)
                # 生成输出文件路径
                output_file_path = os.path.join(output_folder, filename)
                # 保存调整后的图像
                resized_image.save(output_file_path)
                print(f"Resized and saved: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 示例调用
input_folder = 'D:\桌面\水稻叶片病害数据集 Rice Leaf Disease Image Samples\Rice Leaf Disease Images\Tungro'
output_folder = 'D:\桌面\新建文件夹 (2)'
resize_images_in_folder(input_folder, output_folder)