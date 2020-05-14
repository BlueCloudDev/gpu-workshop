import sys, getopt
import torch
import torchvision
import numpy as np
import cv2
import pafy
import urllib.request
from PIL import Image
from dataset import PennFudanDataset, get_transform, get_model_instance_segmentation

def main(argv):
    modelPath = 'model.pth'
    selectedDevice = ''
    source= ''
    youtube = False
    try:
        opts, args = getopt.getopt(argv, "p:d:s:", ["modelPath=", "device=", "source="])
    except getopt.GetoptError:
        print("Error parsing arguments")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-p", "--modelPath"):
            if arg.endswith(".pth") is True or arg.endswith('.pt') is True:
                modelPath = arg
            else:
                print("modelPath must end in .pth or .pt")
                sys.exit(2)
        if opt in ("-d", "--device"):
            lower = arg.lower()
            if lower == "gpu" or lower == "cpu":
                selectedDevice = arg.lower()
            else:
                print("device must be either 'gpu' or 'cpu'")
                sys.exit(2)
        if opt in ("-s", "--source"):
            if arg.startswith("http://") is True or arg.startswith("https://") is True:
                if arg.startswith("https://www.youtube.com") is True or arg.startswith("https://youtube.com") is True or arg.startswith("https://youtu.be") is True:
                    youtube = True
                source = arg
            else:
                print("Source must begin with http:// or https://")
                sys.exit(2)
    
    if source == '':
        print("Please provide a source to evaluate using -s or --source=")
        sys.exit(2)

    # evaluate on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # check if user has requested either cpu or gpu processing
    if torch.cuda.is_available() and selectedDevice == '':
        print("CUDA device is detected, using GPU for training and evaluation")
    elif selectedDevice != '':
        if selectedDevice == 'gpu':
            if torch.cuda.is_available() is False:
                print("Cannot find CUDA driver or device")
                sys.exit(2)
            device = torch.device('cuda')
        if selectedDevice == 'cpu':
            device = torch.device('cpu')

    model = get_model_instance_segmentation(2)
    model.load_state_dict(torch.load(modelPath))
    model.eval()
    model = model.to(device)


    img = None
    if youtube is True:
        vPafy = pafy.new(source)
        play = vPafy.getbest()
        cap = cv2.VideoCapture(play.url)
        _, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
        cv2.destroyAllWindows()
    else:
        with urllib.request.urlopen(source) as url:
            image = np.asarray(bytearray(url.read()), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # You may need to convert the color.
    
    im_pil = Image.fromarray(img).convert("RGB")
    transform = get_transform(train=False)
    im_pil = transform(im_pil, None)
    imList = [im_pil[0]]
    ok = list(image.to(device) for image in imList)
    output = model(ok)

    positives = 0
    weak = 0
    for score in output[0]['scores']:
        if score > 0.99:
            positives = positives + 1
        else:
            weak = weak + 1

    print("---- Results ----")
    if positives == 0:
        print("Did not detect any people")
    else: 
        print("Detected " + str(positives) + " people")
        print("Counted " + str(weak) + " additional weak signals below 99% confidence")


if __name__ == "__main__":
    main(sys.argv[1:])
