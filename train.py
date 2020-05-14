# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import os, sys, getopt
import torch

from engine import train_one_epoch, evaluate
from dataset import PennFudanDataset, get_transform, get_model_instance_segmentation
import utils

def main():
    savePath = 'model.pth'
    selectedDevice = ''
    # let's train it for 10 epochs
    num_epochs = 10
    try:
        opts, args = getopt.getopt(sys.argv[1:], "o:d:e:", ["modelOutputPath=", "device=", "epochs="])
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-o", "--modelOutputPath"):
            if arg.endswith(".pth") is True or arg.endswith('.pt') is True:
                savePath = arg
            else:
                print("Input Error: modelOutputPath must end in .pth or .pt")
                sys.exit(2)
        if opt in ("-d", "--device"):
            lower = arg.lower()
            if lower == "gpu" or lower == "cpu":
                selectedDevice = arg.lower()
            else:
                print("Input Error: Device must be either 'gpu' or 'cpu'")
                sys.exit(2)
        if opt in ("-e", "--epochs"):
            epochs = int(arg)
            if epochs > 0:
                num_epochs = epochs

    # train on the GPU or on the CPU, if a GPU is not available
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


    # our dataset has two classes only - background and person
    num_classes = 2

    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    # these organize the process for sending information to the GPU
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the selected device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # save the model to the current directory 
    torch.save(model.state_dict(), savePath)
    print("Finished")
    
if __name__ == "__main__":
    main()
