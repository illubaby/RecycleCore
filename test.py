#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import multiprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# **KHỞI TẠO MÔ HÌNH**

# In[2]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassification(nn.Module):  
    def training_step(self, batch):  
        images, labels = batch
        out = self(images) 
        loss = F.cross_entropy(out, labels)
        return loss

    def validating(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'Validation Loss': loss.detach(), 'Validation Accuracy': acc}

    def validating_epoch_final(self, outputs):
        batch_loss = [x['Validation Loss'] for x in outputs] 
        # each batch of the validation data
        epoch_loss = torch.stack(batch_loss).mean() 
        batch_accuracy = [x['Validation Accuracy'] for x in outputs]
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'Validation Loss': epoch_loss.item(), 'Validation Accuracy': epoch_accuracy.item()}

    def epoch_final(self, epoch, result):
        print("Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
              .format(epoch + 1, result['Training Loss'], result['Validation Loss'], result['Validation Accuracy']))


# **SỬ DỤNG KIẾN TRÚC ResNet50 CHO VIỆC PHÂN LOẠI**

# In[3]:


class ResNet(ImageClassification):
    def __init__(self):
        super().__init__()
        # Using ResNet50 pretrained model
        self.network = models.resnet50(weights="ResNet50_Weights.DEFAULT")
        features = self.network.fc.in_features
        self.network.fc = nn.Linear(features, len(garbage_classes))

    def forward(self, image):
        return torch.sigmoid(self.network(image))

    def training_step(self, batch):
        images, labels = batch  
        out = self(images) 
        loss = F.cross_entropy(out, labels)
        return loss

    def epoch_final(self, epoch, result):
        print("Epoch [{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}"
              .format(epoch + 1, result['Train Loss'], result['Validation Loss'], result['Validation Accuracy']))


# **THIẾT LẬP HÀM XUẤT CÁC THÔNG SỐ KẾT QUẢ**

# In[4]:


def export_classification_metrics(model, dataloader, classes):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = move_to_gpu(images, device)
            labels = labels.to(device)  
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy()) 

    for i, class_name in enumerate(classes):
        class_labels = [1 if label == i else 0 for label in all_labels]
        class_preds = [1 if pred == i else 0 for pred in all_preds]
        
        accuracy = accuracy_score(class_labels, class_preds) * 100
        precision = precision_score(class_labels, class_preds, zero_division=0) * 100
        recall = recall_score(class_labels, class_preds, zero_division=0) * 100
        f1 = f1_score(class_labels, class_preds, zero_division=0) * 100
        
        print(f'{class_name}, accuracy: {accuracy:.2f}%, precision: {precision:.2f}%, recall: {recall:.2f}%, F1 score: {f1:.2f}%')


# **THIẾT LẬP EARLY STOPPPING**

# In[5]:


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, restore_best_weights=True):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.restore_best_weights = restore_best_weights
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_weights = model.state_dict().copy()
            if self.verbose:
                print(f'Validation loss improved: {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not improve: {val_loss:.4f}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_model_weights is not None:
                    model.load_state_dict(self.best_model_weights)
                    if self.verbose:
                        print("Restored best model weights.")


# **THIẾT LẬP HÀM ĐO HIỆU SUẤT CHÍNH XÁC GIỮA CÁC LOẠI RÁC**

# In[6]:


def plot_accuracy_per_class(model, dataloader, classes):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = move_to_gpu(images, device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(labels.cpu().numpy())

    accuracies = []
    for i, class_name in enumerate(classes):
        class_labels = [1 if label == i else 0 for label in all_labels]
        class_preds = [1 if pred == i else 0 for pred in all_preds]

        accuracy = accuracy_score(class_labels, class_preds) * 100
        accuracies.append(accuracy)

    plt.bar(classes, accuracies)
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Class')
    plt.xticks(rotation=45)
    plt.show()


# **LOAD BỘ DỮ LIỆU TRASHNET**

# In[7]:


directory = 'trashnet/dataset-resized'
global garbage_classes
garbage_classes = os.listdir(directory)
print(garbage_classes)


# **BIẾN ĐỔI BỘ DỮ LIỆU**

# In[8]:


# Import Transforms để chỉnh sửa ảnh
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# Resize ảnh về kích thước 224x224 pixels và transform thành Tensor
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Load bộ dữ liệu và áp dụng transformations
dataset = ImageFolder(directory, transform=transformations)


# **HIỂN THỊ ẢNH NGẪU NGHIÊN VỚI LABEL ĐỂ TEST BỘ DỮ LIỆU**

# In[9]:


def display_test(image, label):
    print("Label:", dataset.classes[label], "(Class No: " + str(label) + ")")
    plt.imshow(image.permute(1, 2, 0))
    plt.show()

# Hiển thị ảnh ngẫu nhiên từ bộ dữ liệu
image, label = dataset[random.randint(0, len(dataset))]
display_test(image, label)


# **THIẾT LẬP SEED NGẪU NHIÊN PHỤC VỤ CHO VIỆC TÁI TẠO**

# In[10]:


random_seed = 43
torch.manual_seed(random_seed)


# **CHIA BỘ DỮ LIỆU & THIẾT LẬP BATCH SIZE**

# In[11]:


# Chia bộ dữ liệu thành các tập train, validate, và test
train_size = int(0.6 * len(dataset))  # 60% bộ dữ liệu
val_size = int(0.2 * len(dataset))  # 20% bộ dữ liệu
test_size = len(dataset) - train_size - val_size  # 20% bộ dữ liệu còn lại
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
from torch.utils.data.dataloader import DataLoader
batch_size = 64


# **THIẾT LẬP DATALOADERS**

# In[12]:


train = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
validation = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)


def main():
    # Optional: Support for freezing on Windows
    from multiprocessing import freeze_support
    freeze_support()

    # ------------------ Setup ------------------ #
    # Set up device
    device = get_default_device()
    print('Processing via:', device)
    
    # Set up DataLoaders (using your already defined dataset split)
    from torch.utils.data.dataloader import DataLoader
    batch_size = 64
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(val_data, batch_size * 2, num_workers=4, pin_memory=True)
    
    # Visualize one batch (this uses the DataLoader directly)
    batch_visualization(train_loader)
    
    # Initialize model and move it to the GPU/CPU device
    model = ResNet()
    model = move_to_gpu(model, device)
    
    # Wrap the DataLoaders with your custom DataLoad class (which moves data to the device)
    train_data_loader = DataLoad(train_loader, device)
    validation_data_loader = DataLoad(validation_loader, device)
    
    # ------------------ Evaluate & Train ------------------ #
    # (Optional) Evaluate the initial model on validation data
    evaluate(model, validation_data_loader)
    
    # Set hyperparameters and start training
    epochs = 50
    learning_rate = 0.00005
    patience = 3
    optimizer = torch.optim.Adam  # You can change this if needed
    model_history = opt(epochs, learning_rate, model, train_data_loader, validation_data_loader, optimizer, patience)
    
    # ------------------ Plot Results ------------------ #
    plot_accuracy(model_history)
    plot_loss(model_history)
    
    # ------------------ Export Metrics & Confusion Matrix ------------------ #
    export_classification_metrics(model, validation_data_loader, garbage_classes)
    plot_accuracy_per_class(model, validation_data_loader, garbage_classes)
    plot_confusion_matrix(model, validation_data_loader, garbage_classes)
    
    # ------------------ Testing Predictions ------------------ #
    # Testing No.1
    img, label = random.choice(dataset)
    plt.imshow(img.permute(1, 2, 0))
    print('Testing No.1 - Class:', dataset.classes[label], ', Predicted Class:', predict(img, model))
    plt.title('Testing No.1')
    plt.show()
    
    # You can add Testing No.2 and No.3 similarly if desired.
    
    # ------------------ Save the Model ------------------ #
    FILE = "/kaggle/working/ResnetModel.pth"
    model_scripted = torch.jit.script(model)
    torch.jit.save(model_scripted, FILE)

