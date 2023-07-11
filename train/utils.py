
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
    
    for sub in subdirectories:
        os.mkdir(os.path.join(sub, "g"))
        os.mkdir(os.path.join(sub, "class"))
        os.mkdir(os.path.join(sub, "code"))
        os.mkdir(os.path.join(sub, "other"))
        os.mkdir(os.path.join(sub, "chat"))
        chat_id = int(os.path.basename(sub))
        print(chat_id)
        for root, dirs, files in os.walk(sub):
            for file in files:
                # Perform operations on each file
                file_path = os.path.join(root, file)
                print(file_path)
                # Your code here
                if callback:
                    callback(file_path)

# Call the function with the desired directory
# recursively_iterate_dir("/data/vps/inference",callback=None)

move_file("/data/vps/inference/1733122226/I","g")