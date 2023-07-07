
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