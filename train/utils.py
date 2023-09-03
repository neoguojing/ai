
import torch
import csv

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def write_csv(row):
    with open("alber_infence.csv", "w", newline="") as csvfile:
        # Create the CSV writer object
        writer = csv.writer(csvfile)
        writer.writerow(row)

def read_csv():
    with open("alber_infence.csv", newline="") as csvfile:
        # Create the CSV reader object
        reader = csv.reader(csvfile)

        # Loop over the rows of the file
        for row in reader:
            print(row)

def ocr(image_path):
   import pytesseract
   from PIL import Image

   # Load the image
   image = Image.open(image_path)

   # Perform OCR
   text = pytesseract.image_to_string(image, lang='chi_sim')
   print(text)
   # Print the extracted text
   return text


import os
import shutil

def move_file(image_path,predicted_class):
    dir_name = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    dst_path = os.path.join(dir_name, predicted_class,file_name)
    shutil.move(image_path,dst_path)

def get_subdirectories(path):
    """
    获取指定目录下的所有子目录
    """
    subdirectories = []
    for name in os.listdir(path):
        dir_path = os.path.join(path, name)
        if os.path.isdir(dir_path):
            subdirectories.append(dir_path)
    return subdirectories

def recursively_iterate_dir(directory,callback=None):
    subdirectories = get_subdirectories(directory)
    print(subdirectories)
    count = 0
    for sub in subdirectories:
        count += 1
        os.makedirs(os.path.join(sub, "g"), exist_ok=True)
        os.makedirs(os.path.join(sub, "class"), exist_ok=True)
        os.makedirs(os.path.join(sub, "code"), exist_ok=True)
        os.makedirs(os.path.join(sub, "other"), exist_ok=True)
        os.makedirs(os.path.join(sub, "chat"), exist_ok=True)
        chat_id = int(os.path.basename(sub))
        print(chat_id)
        for f in os.listdir(sub):
            file_path = os.path.join(sub, f)
            if os.path.isfile(os.path.join(sub, f)):
                print(file_path)
                if f.endswith(".jpeg") or f.endswith(".jpg"):                
                    # Your code here
                    if callback:
                        callback(file_path)
        print("curent count:",count)
# Call the function with the desired directory
# recursively_iterate_dir("/data/vps/inference",callback=None)

def patch_empty_dir(root_path):
    # 遍历目录树，查找空目录
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                print(f"{dir_path} is an empty directory.")
                print(f"{dir}")
                if dir == "g":
                    shutil.copy("./patch/g.jpeg", dir_path)
                elif dir == "chat":
                    shutil.copy("./patch/chat.jpeg", dir_path)
                elif dir == "code":
                    shutil.copy("./patch/code.jpeg", dir_path)
                elif dir == "class":
                    shutil.copy("./patch/class.jpeg", dir_path)
                elif dir == "other":
                    shutil.copy("./patch/other.jpeg", dir_path)

# patch_empty_dir("/data/vps/train")