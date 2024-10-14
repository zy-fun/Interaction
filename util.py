from PIL import Image
import os

# 图片文件夹路径
image_folder = 'gnnfig'

# 获取图片文件列表
images = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]

# 打开所有图片并添加到列表中
frames = []
for image in images:
    new_frame = Image.open(image)
    frames.append(new_frame)

# 保存为 GIF 动图
frames[0].save(os.path.join(image_folder, 'gnn.gif'), format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=100,  # 每帧的显示时间，单位为毫秒
               loop=0)       # 循环次数，0 表示无限循环