# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import os
import torch

from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

def process_image(input_path, output_path, model):
    file_client = FileClient('disk')
    img_bytes = file_client.get(input_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception(f"Path {input_path} not working")

    img = img2tensor(img, bgr2rgb=True, float32=True)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})
    model.test()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    imwrite(sr_img, output_path)
    print(f"Inference {input_path} .. finished. Saved to {output_path}")

def process_folder(input_folder, output_folder, model):
    # 如果輸出資料夾不存在，則創建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 讀取輸入資料夾中的所有圖片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path, model)

def main():
    # parse options, set distributed setting, set random seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    input_folders = ["./adjustsize/uav0000124_00944_v/", "./adjustsize/uav0000126_00001_v/", "./adjustsize/uav0000138_00000_v/", "./adjustsize/uav0000140_01590_v/", "./adjustsize/uav0000143_02250_v/", "./adjustsize/uav0000145_00000_v/", "./adjustsize/uav0000150_02310_v/", "./adjustsize/uav0000218_00001_v/", "./adjustsize/uav0000222_03150_v/", "./adjustsize/uav0000266_03598_v/", "./adjustsize/uav0000266_04830_v/", "./adjustsize/uav0000295_02300_v/", "./adjustsize/uav0000316_01288_v/"]  # 輸入資料夾路徑列表
    output_folders = ['./deblurvisdrone/uav0000124_00944_v/', './deblurvisdrone/uav0000126_00001_v/', './deblurvisdrone/uav0000138_00000_v/', './deblurvisdrone/uav0000140_01590_v/', './deblurvisdrone/uav0000143_02250_v/', './deblurvisdrone/uav0000145_00000_v/', './deblurvisdrone/uav0000150_02310_v/', './deblurvisdrone/uav0000218_00001_v/', './deblurvisdrone/uav0000222_03150_v/', './deblurvisdrone/uav0000266_03598_v/', './deblurvisdrone/uav0000266_04830_v/', './deblurvisdrone/uav0000295_02300_v/', './deblurvisdrone/uav0000316_01288_v/']  # 輸出資料夾路徑列表

    model = create_model(opt)
    #model.eval()

    # 逐一處理每個輸入資料夾
    for input_folder, output_folder in zip(input_folders, output_folders):
        process_folder(input_folder, output_folder, model)

if __name__ == '__main__':
    main()

