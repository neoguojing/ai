# Import necessary libraries
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from crnn import crnn
from dbnet.inference import DBNet

# Define function for OCR with DBNet+CRNN
def ocr(image_path, dbnet_model_path, crnn_model_path):
    # Load DBNet model
    dbnet = DBNet(dbnet_model_path)
    # Load CRNN model
    crnn_net = crnn.CRNN(32, 1, 37, 256, 1).cuda()
    crnn_net.load_state_dict(torch.load(crnn_model_path))
    converter = crnn.utils.strLabelConverter('0123456789abcdefghijklmnopqrstuvwxyz')
    # Read image
    img = cv2.imread(image_path)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect text regions using DBNet
    boxes, _ = dbnet.detect(img)
    # Sort boxes from left to right
    boxes = sorted(boxes, key=lambda x: x[0][0])
    # Initialize results string
    results = ''
    # Loop over each box
    for box in boxes:
        # Extract box coordinates
        x1, y1, x2, y2, x3, y3, x4, y4 = box[0]
        # Warp perspective to extract text region
        warp = dbnet.warpPerspective(gray, box)
        # Resize image to match CRNN input size
        warp = cv2.resize(warp, (100, 32))
        # Convert image to tensor
        warp = warp.astype(np.float32)
        warp = warp / 255.0
        warp = warp.transpose([2, 0, 1])
        warp = torch.from_numpy(warp)
        warp = warp.unsqueeze(0)
        # Pass image through CRNN
        warp = Variable(warp).cuda()
        preds = crnn_net(warp)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        # Convert CRNN output to string
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        # Append result to results string
        results += sim_pred
    # Return results string
    return results