import collections
import logging
import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Classifier01, Classifier02, Classifier03, Classifier04, Classifier05

def get_image_pixel_stats(image: float, channel_num: int) -> tuple[float, float, float, float]:
    """
    Generate aggregate stats on pixel values for the input image (i.e. min, mean, median, max).

    Args:
        image: torch tensor for image.  expected image.shape = (1, num_channels, height, width) = (1, 3, 32, 32)

    Return:
        min_pixel_val, mean_pixel_val, median_pixel_val, max_pixel_val
    """
    # NOTE: this implementation is modified from my solution to the project for
    #       "AI programming with Python"
    #       https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/exploration%20-%20inspecting%20data.ipynb
    num_channels = image.shape[1]
    if channel_num > num_channels - 1:
        min_pixel_val = None
        mean_pixel_val = None
        median_pixel_val = None
        max_pixel_val = None
    else:
        min_pixel_val = image[0,channel_num].min().item()
        mean_pixel_val = image[0,channel_num].mean().item()
        median_pixel_val = image[0,channel_num].median().item()
        max_pixel_val = image[0,channel_num].max().item()
    return min_pixel_val, mean_pixel_val, median_pixel_val, max_pixel_val


def get_image_stats(image: torch.Tensor) -> tuple[list[Union[float,int]], list[str]]:
    """
    Generate aggregate stats for an input image (i.e. num channels, hieght, width, 
    area, aspect ratio, and aggregate stats on pixel values

    Args:
        image: torch tensor for image.  expected image.shape = (1, num_channels, height, width) 
        = (1, 3, 32, 32)
    """
    # NOTE: this implementation is modified from my solution to the project for
    #       "AI programming with Python"
    #       https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/exploration%20-%20inspecting%20data.ipynb
    # expected image.shape: (1, num_channels=3, height, width)
    num_channels = image.shape[1]
    height, width = tuple(image.shape[2:])
    area = height * width
    aspect_ratio = width / height

    row = [num_channels, height, width, area, aspect_ratio]

    # add pixel summary stats from each channel to row
    for channel_num in range(3):
        min_pixel_val, mean_pixel_val, median_pixel_val, max_pixel_val = get_image_pixel_stats(image, channel_num)
        row.extend([min_pixel_val, mean_pixel_val, median_pixel_val, max_pixel_val])
    
    columns = ["num_channels", "height", "width", "area", "aspect_ratio"]
    for channel_num in range(3):
        columns.extend([f"chan_{channel_num}_min_pixel_val", f"chan_{channel_num}_mean_pixel_val", 
                        f"chan_{channel_num}_median_pixel_val", f"chan_{channel_num}_max_pixel_val"])

    return row, columns


def get_class_idx_to_class_name_mapping(class_name_to_class_idx):
    """
    """
    class_idx_to_class_name = dict()
    for class_name, class_idx in class_name_to_class_idx.items():
        if class_idx in class_idx_to_class_name:
            raise AssertionError(f"class_idx to class_name mapping is not one to one.  class_idx ({class_idx}) repeated")
        class_idx_to_class_name[class_idx] = class_name
    return class_idx_to_class_name


def get_cifar10_image_stats(root: str = "./cifar10-data", image_count: int = None):
    """
    Collect image stats for CIFAR-10 training set.

    Args:
        root: root directory to which CIFAR10 data will be downloaded
        image_count: the number of images to process.  If None, then process all.

    Return:
        df_image_stats, df_label_stats
    """
    # NOTE: this implementation is modified from my solution to the project for
    #       "AI programming with Python"
    #       https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/exploration%20-%20inspecting%20data.ipynb    
    if image_count is not None and image_count <= 0:
        raise ValueError(f"image_count ({image_count}) must be > 0")

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    dataloader = DataLoader(trainset, batch_size=1, shuffle=False)

    if type(trainset) == torch.utils.data.dataset.Subset:
        class_name_to_class_idx = trainset.dataset.class_to_idx
    else:
        class_name_to_class_idx = trainset.class_to_idx
    class_idx_to_class_name = get_class_idx_to_class_name_mapping(class_name_to_class_idx)

    # process one image at a time
    image_data = []
    class_idx_counter = collections.Counter()
    for image_num, (image, class_idx) in enumerate(dataloader):
        if image_count is not None and image_num == image_count:
            break
        row, columns = get_image_stats(image)
        image_data.append(row)

        class_idx_counter[class_idx.item()] += 1

    # build image stats dataframe
    df_image_stats = pd.DataFrame(image_data, columns=columns)
    
    # build label stats dataframe
    label_data = [[class_idx, class_idx_to_class_name[class_idx], count] 
                        for class_idx, count in class_idx_counter.items()]
    label_data.sort(key = lambda row: row[0])
    df_label_stats = pd.DataFrame(label_data, columns=["class_idx", "class_name", "count"])

    return df_image_stats, df_label_stats


def get_model_performance_stats(model, criterion, loader, device):
    """
    """
    # NOTE: this implementation is modified from my solution to the project for
    #       "AI programming with Python"
    #        https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/utils.py
    start_time = time.time()
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    model.eval()
    for inputs, labels in loader:
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device=device), labels.to(device=device)

        # forward propagate (no gradient tracking)
        with torch.no_grad():
            logps = model.forward(inputs)

        # calculate loss and add to running total
        loss = criterion(logps, labels)
        running_loss += loss.item()

        # determine accuracy and add to running totals
        torch.topk(logps, 1, dim=1).values
        num_correct += (torch.topk(logps, 1, dim=1).indices == labels.view(-1,1)).sum().item()
        num_total += len(labels)
    model.train()
    loss = running_loss / len(loader)
    accuracy = num_correct / num_total

    run_time = time.time() - start_time

    return loss, accuracy, run_time


def get_accuracy(output, labels):
    """
    Get accuracy from forward propagation output and labels.
    """
    # NOTE: this is a copy and past of my solution to the project for
    #       "AI programming with Python"
    #        https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/utils.py
    num_correct = (torch.topk(output, 1, dim=1).indices == labels.view(-1,1)).sum().item()
    num_total = len(labels)
    accuracy = num_correct / num_total
    return accuracy


def train_epoch(model, criterion, optimizer, train_dataloader, device):
    """
    
    """
    # NOTE: this implementation is modified from my solution to the project for
    #       "AI programming with Python"
    #        https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/utils.py
    num_batches = len(train_dataloader)
    losses = np.zeros(num_batches, dtype=np.double)
    batch_sizes = np.zeros(num_batches, dtype=int)
    accuracies = np.zeros(num_batches, dtype=np.double)
    start_time = time.time()
    for batch_num, (inputs, labels) in enumerate(train_dataloader):
        # # print batch number periodically to show progress
        # if ((batch_num + 1) % print_every_n_batches == 0) or (batch_num == len(self.training_data_loader) - 1):
        #     logging.info(f"\tbatch_num: {batch_num + 1} / {len(self.training_data_loader)}")
        
        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device=device), labels.to(device=device)

        # forward propagate and calculate loss
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        accuracy = get_accuracy(outputs, labels)

        # backpropagate and take optimization step
        # gradients are zeroed before backpropagation since they accumulate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep track of stats
        batch_size = len(labels)
        losses[batch_num] = loss.item()
        accuracies[batch_num] = accuracy
        batch_sizes[batch_num] = batch_size

    # NLL loss by default uses a "mean" reduction.  I will average over batch and not worry
    # if the last batch isn't the same size
    avg_loss = losses.sum() / num_batches

    # Overall accuracy is most appropriately calculated by a weighted average over batch sizes.
    avg_accuracy = (accuracies * batch_sizes).sum() / batch_sizes.sum()

    run_time = time.time() - start_time

    return avg_loss, avg_accuracy, run_time


def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device):
    """
    """
    # NOTE: this implementation is modified from my solution to the project for
    #       "AI programming with Python"
    #        https://github.com/mrperkett/udacity-project-create-image-classifier/blob/main/aipnd-project/utils.py

    # move model tensors to device
    model.to(device=device)

    stat_names = ["training_avg_losses", 
                  "training_avg_accuracies",
                  "training_run_times",
                  "validation_losses",
                  "validation_accuracies",
                  "validation_run_times",
                  "epoch_run_times"]
    stats_dict = {stat_name : [] for stat_name in stat_names}

    header_line = "".join([val.ljust(12) for val in ["epoch", "run_time", "train_loss", "train_acc", "valid_loss", "valid_acc"]])
    logging.info(header_line)
    start_time = time.time()
    for epoch in range(num_epochs):
        # run training for a single epoch
        avg_training_loss, avg_training_accuracy, training_run_time = \
                train_epoch(model, criterion, optimizer, train_dataloader, device)

        # calculate validation loss and accuracy
        validation_loss, validation_accuracy, validation_run_time = \
                get_model_performance_stats(model, criterion, validation_dataloader, device=device)
        

        # save stats
        epoch_run_time = training_run_time + validation_run_time
        stats_dict["training_avg_losses"].append(avg_training_loss)
        stats_dict["training_avg_accuracies"].append(avg_training_accuracy)
        stats_dict["training_run_times"].append(training_run_time)
        stats_dict["validation_losses"].append(validation_loss)
        stats_dict["validation_accuracies"].append(validation_accuracy)
        stats_dict["validation_run_times"].append(validation_run_time)
        stats_dict["epoch_run_times"].append(epoch_run_time)

        formatted_stats = [f"{epoch}",
                           f"{epoch_run_time:.1f}", 
                           f"{avg_training_loss:.4f}",
                           f"{avg_training_accuracy:.4f}",
                           f"{validation_loss:.4f}",
                           f"{validation_accuracy:.4f}"]
        stats_line = "".join([val.ljust(12) for val in formatted_stats])
        logging.info(stats_line)

    total_run_time = time.time() - start_time

    logging.info(f"total_run_time: {total_run_time}")

    return stats_dict


def plot_loss_during_training(training_stats_dict: dict[str, list[float]]):
    """
    Plot training loss and validation loss by epoch number.

    Args:
        training_stats_dict: dictionary with stat_name (string) as the key and
        a list of statistic values (floats) as the value.  This function requires
        the "training_avg_losses" and "validation_losses" keys.
    
    Return:
        ax: matplotlib axis
    """
    num_tick_marks = 8
    _, ax = plt.subplots()

    num_epochs = len(training_stats_dict["training_avg_losses"])
    ax.plot(training_stats_dict["training_avg_losses"], label="Training")
    ax.plot(training_stats_dict["validation_losses"], label="Validation")
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.set_title("Loss during training", fontsize=22)
    ax.set_xticks(np.arange(0, num_epochs, num_epochs // num_tick_marks))
    ax.legend()

    return ax


def plot_accuracy_during_training(training_stats_dict: dict[str, list[float]]):
    """
    Plot training accuracy and validation accuracy by epoch number.

    Args:
        training_stats_dict: dictionary with stat_name (string) as the key and
        a list of statistic values (floats) as the value.  This function requires
        the "training_avg_accuracies" and "validation_accuracies" keys.
    
    Return:
        ax: matplotlib axis
    """
    num_tick_marks = 8
    _, ax = plt.subplots()

    num_epochs = len(training_stats_dict["training_avg_accuracies"])
    ax.plot(training_stats_dict["training_avg_accuracies"], label="Training")
    ax.plot(training_stats_dict["validation_accuracies"], label="Validation")
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Loss", fontsize=18)
    ax.set_title("Validation accuracy during training", fontsize=22)
    ax.set_xticks(np.arange(0, num_epochs, num_epochs // num_tick_marks))
    ax.legend()

    return ax


def save_checkpoint(model: nn.Module, dataset, training_stats_dict, output_file_path: str) -> None:
    """
    Save model to checkpoint file with enough information to restore for inference or to
    continue training.

    Args:
        model: classifier neural network module to be saved
        dataset: PyTorch Dataset - used to store class idx and class name mappings
        training_stats_dict: dictionary containing stats from during training (returned
                             by train())
        output_file_path: file path to which checkpoint file will be written
    
    Return:
        None
    """
    # get the name of the class
    class_name_of_model = model._get_name().split(".")[-1]

    # get class idx and name mapping dictionaries
    if type(dataset) == torch.utils.data.dataset.Subset:
        class_name_to_class_idx = dataset.dataset.class_to_idx
    else:
        class_name_to_class_idx = dataset.class_to_idx
    class_idx_to_class_name = get_class_idx_to_class_name_mapping(class_name_to_class_idx)

    # save to checkpoint file
    checkpoint_dict = {"class_name_of_model" : class_name_of_model,
                       "state_dict" : model.state_dict(),
                       "training_stats_dict" : training_stats_dict,
                       "class_name_to_class_idx" : class_name_to_class_idx,
                       "class_idx_to_class_name" : class_idx_to_class_name}
    torch.save(checkpoint_dict, output_file_path)

    return


def load_checkpoint(checkpoint_file_path: str) -> nn.Module:
    """
    Load saved model from checkpoint file.

    Args:
        checkpoint_file_path: file path to checkpoint file to load
    """
    # read checkpoint file into dictionary
    checkpoint_dict = torch.load(checkpoint_file_path)

    # Initialize model
    class_name_of_model = checkpoint_dict["class_name_of_model"]
    model = eval(f"{class_name_of_model}()")

    # load all tensors into model
    model.load_state_dict(checkpoint_dict["state_dict"])

    # remove the keys that are no longer necessary from checkpoint_dict before returning it
    checkpoint_dict.pop("class_name_of_model")
    checkpoint_dict.pop("state_dict")

    return model, checkpoint_dict