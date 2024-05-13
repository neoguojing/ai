# detection2的数据集如何使用，针对每个数据集编写数据集下载代码
import requests
import zipfile
import os

def download_dataset():
    # 设置URL
    url = 'https://example.com/detection2.zip'  # 请替换为实际的数据集链接
    dataset_name = 'detection2'
    data_folder = 'data'

    # 创建数据文件夹
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # 下载数据集
    zip_path = os.path.join(data_folder, f'{dataset_name}.zip')
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    # 解压数据集
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_folder)

    # 删除压缩文件
    os.remove(zip_path)

    print(f'{dataset_name} 数据集已成功下载到 {data_folder} 文件夹中.')

# 调用函数下载数据集
download_dataset()
