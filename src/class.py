import torch
import sys
sys.path.insert(0, '')
from  model_factory import ModelFactory
from tools import image_preprocessor
from dataset import imagenet_labels

# Define the classification function for multi-class classification
def classification(image_path, model_name):
    # Get the model
    model = ModelFactory.create_classication_model(model_name)

    # Preprocess the image
    input_batch = image_preprocessor(image_path)

    if torch.cuda.is_available():
        input_batch = input_batch.cuda()
        print("using gpu")

    with torch.no_grad():
        output = model(input_batch)

    print(output[0])
    # Post-process the output
    label = post_processor(output)
    return label



# Define the post-processor function to convert class result to human read format
def post_processor(output):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    output = []
    for i in range(top5_prob.size(0)):
        print(imagenet_labels[top5_catid[i]], top5_prob[i].item())
        output.append({imagenet_labels[top5_catid[i]]:top5_prob[i].item()})
    # # Get the index of the predicted class
    # _, index = torch.max(output, 1)
    # # Convert the index to a human-readable label
    # label = imagenet_labels[index[0]]
    return output


