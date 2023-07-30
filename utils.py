import numpy as np
import pandas as pd
import numpy as np
import nibabel as nib
import glob
import torch
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from earlystop import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, multi_class='ovr', average=average)


def class_vs_rest_roc_auc_score(y_test, y_pred):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)
    y_pred_bin = lb.transform(y_pred)
    roc_auc_scores = {}

    for i, class_label in enumerate(lb.classes_):
        class_indices = lb.transform([class_label])[0]
        rest_indices = ~class_indices
        y_test_class = y_test_bin[:, i]
        y_test_rest = y_test_bin[:, rest_indices]
        y_pred_class = y_pred_bin[:, i]
        y_pred_rest = y_pred_bin[:, rest_indices]
        roc_auc_scores[class_label] = roc_auc_score(y_test_class, y_pred_class, multi_class='ovr', average="macro")

    return roc_auc_scores

def calculate_specificity(cm):
    num_classes = cm.shape[0]
    specificity = {}

    for i in range(num_classes):
        tp = cm[i, i]
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        if tn + fp == 0:
            specificity[i] = 0

        specificity[i] = 100*(tn / (tn + fp))

        if specificity[i] == 'nan':
            specificity[i] = 0

    return specificity

def calculate_sensitivity(cm):
    num_classes = cm.shape[0]
    sensitivity = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp

        if tp + fn == 0:
            sensitivity[i] = 0

        sensitivity[i] = 100*(tp / (tp + fn))

        if sensitivity[i] == 'nan':
            sensitivity[i] = 0

    return sensitivity


def calculate_precision(cm):
    num_classes = cm.shape[0]
    precision = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp

        if tp + fp == 0:
            precision[i] = 0
        else:
            precision[i] = 100 * (tp / (tp + fp))

        if np.isnan(precision[i]):
            precision[i] = 0

    return precision

def calculate_f1_score(cm):
    num_classes = cm.shape[0]
    f1_scores = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        if tp + fp == 0 or tp + fn == 0:
            f1_scores[i] = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            if np.isnan(precision) or np.isnan(recall):
                f1_scores[i] = 0
            else:
                f1_scores[i] = (2 * precision * recall) / (precision + recall) * 100

    return f1_scores

def calculate_accuracy(cm):
    num_classes = cm.shape[0]
    accuracy = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn

        if tp + tn + fp + fn == 0:
            accuracy[i] = 0

        accuracy[i] = 100 * ((tp + tn) / (tp + tn + fp + fn))

        if accuracy[i] == 'nan':
            accuracy[i] = 0

    return accuracy


def flatten(lst):
    flattened = []
    for sublist in lst:
        flattened.extend(sublist)
    return flattened


def slices(vol):
    images = []
    for i in range(1, 8):
        input_img = vol[:, :, i * 6 - 1]
        images.append(input_img)
    processed_img = np.array(images)
    return processed_img

def slices3d(vol):
    images = []
    #print('eto que e lo e manito', vol.shape)
    for i in range(1, 8):
        input_img = vol[:, :, i * 6 - 1]
        #print('This is indeed the input my men', input_img.shape)
        images.append(input_img)
    processed_img = np.array(images)
    return processed_img
def slices_RESNET(vol):
    images = []
    for i in range(1, 8):
        input_img = vol[:, :, i * 6 - 1]
        resized_img = resize(input_img, (224, 224), anti_aliasing=True)
        images.append(resized_img)
    processed_img = np.array(images)
    return processed_img

def slices_MC(vol):
    images = []
    for i in range(1, 8):
        input_img = vol[:, :, i * 6 - 1]
        resized_img = resize(input_img, (128, 171), anti_aliasing=True)
        images.append(resized_img)
    processed_img = np.array(images)
    return processed_img

#class DataLoader2DT1GD():
    #def __init__(self, img_dir=r'/data/projects/TMOR/data/VeryFinalA/', label_file=r'final_labels.xlsx'):
        #self.all_items = glob.glob(os.path.join(img_dir, '*', '*'))
        #self.n_items = len(self.all_items)
        #self.all_y = pd.read_excel(label_file)['labels']

    #def __getitem__(self, indices: list) -> torch.Tensor:
        #volumes = []
        #for i in indices:
            #vols = []
            #path = os.path.join(self.all_items[i], f't1_gd.nii.gz')
            #vol = nib.load(path).get_fdata()
            #slice = slices(vol)
            #vols.append(slice[3])
            #volumes.append(np.stack(vols))
        #tensor = np.stack(volumes) / 255
        #X = torch.Tensor(tensor)
        #y = [self.all_y[index] for index in indices]
        #return X, y


class DataLoader2DT1GD():
    def __init__(self, img_dir=r'/data/projects/TMOR/data/VeryFinalA/', label_file=r'final_labels.xlsx'):
        self.all_items = glob.glob(os.path.join(img_dir, '*', '*'))
        self.n_items = len(self.all_items)
        self.all_y = pd.read_excel(label_file)['labels']

    def __getitem__(self, indices: list) -> torch.Tensor:
        volumes = []
        for i in indices:
            vols = []
            path = os.path.join(self.all_items[i], f't1_gd.nii.gz')
            vol = nib.load(path).get_fdata()
            slice = slices_RESNET(vol)
            vols.append(slice[3])
            volumes.append(np.stack(vols))
        tensor = np.stack(volumes)
        X = torch.Tensor(tensor)
        #X = np.array(vols)
        y = [self.all_y[index] for index in indices]
        return X, y

class DataLoader3DT1GD():
    def __init__(self, img_dir=r'/data/projects/TMOR/data/VeryFinalA/', label_file=r'final_labels.xlsx'):
        self.all_items = glob.glob(os.path.join(img_dir, '*', '*'))
        self.n_items = len(self.all_items)
        self.all_y = pd.read_excel(label_file)['labels']

    def __getitem__(self, indices: list) -> torch.Tensor:
        volumes = []
        for i in indices:
            vols = []
            path = os.path.join(self.all_items[i], f't1_gd.nii.gz')
            vol = nib.load(path).get_fdata()
            #slice = slices_MC(vol)
            #vols.append(slice)
            #volumes.append(np.stack(vols))
            volumes.append(vol)
        #tensor = np.stack(volumes)
        #X = torch.Tensor(tensor)
        volumes = np.array(volumes)
        X = torch.Tensor(volumes)
        y = [self.all_y[index] for index in indices]
        return X, y

class DataLoader2DALL():
    def __init__(self, img_dir=r'/data/projects/TMOR/data/VeryFinalA/', label_file=r'final_labels.xlsx'):
        self.all_items = glob.glob(os.path.join(img_dir, '*', '*'))
        self.n_items = len(self.all_items)
        self.all_y = pd.read_excel(label_file)['labels']

    def __getitem__(self, indices: list) -> torch.Tensor:
        volumes = []
        for i in indices:
            vols = []
            for file in 'seg', 't1_gd', 't1_pre':
                path = os.path.join(self.all_items[i], f'{file}.gz')
                vol = nib.load(path).get_fdata()
                slice = slices_RESNET(vol)
                vols.append(slice[3])
            volumes.append(np.stack(vols))
        tensor = np.stack(volumes)
        X = torch.Tensor(tensor)
        y = [self.all_y[index] for index in indices]
        return X, y

class DataLoader2():
    def __init__(self, img_dir=r'/data/projects/TMOR/data/VeryFinalA/', label_file=r'final_labels.xlsx'):
        self.all_items = glob.glob(os.path.join(img_dir, '*', '*'))
        self.n_items = len(self.all_items)
        self.all_y = pd.read_excel(label_file)['labels']

    def __getitem__(self, indices: list) -> torch.Tensor:
        volumes = []
        for i in indices:
            vols = []
            for file in 'seg', 't1_gd', 't1_pre':
                path = os.path.join(self.all_items[i], f'{file}.gz')
                vols.append(nib.load(path).get_fdata())
            volumes.append(np.stack(vols))
        tensor = np.stack(volumes)
        X = torch.Tensor(tensor)
        y = [self.all_y[index] for index in indices]
        return X, y


def visualize_slices(img_dir):
    # Load the image volume
    vol = nib.load(img_dir).get_fdata()

    # Process the images
    processed_images = slices(vol)

    # Visualize the slices
    num_slices = processed_images.shape[0]
    fig, axs = plt.subplots(1, num_slices, figsize=(15, 5))
    for i in range(num_slices):
        axs[i].imshow(processed_images[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"Slice {i+1}")

    plt.tight_layout()
    plt.show()

#class CustomDataset(Dataset):
    #def __init__(self, X, y=None):
        #self.X = X
        #self.y = y

    #def __len__(self):
        #return len(self.X)

    #def __getitem__(self, index):
        #x_item = self.X[index]
        #if self.y is not None:
            #y_item = self.y[index]
            #return x_item, y_item
        #else:
            #return x_item


class CustomDataset3D(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        #print(x.shape)
        y = self.y[index]

        volumes = []
        #if self.transform:
            #vols = []
            #x_pil = transforms.ToPILImage()(x)
            #x_pill_trans = self.transform(x)
            #x = x_pill_trans
            #x = transforms.ToTensor()(x_pill_trans)
            #slice = slices(x_pill_trans)
            #vols.append(slice)
            #x = torch.Tensor(vols)
        #else:
            #vols = []
            #x_pil = transforms.ToPILImage()(x)
            #x_pill_trans = self.transform(x)
            # x = x_pill_trans
            #x = transforms.ToTensor()(x_pill_trans)
            #slice = slices(x)
            #vols.append(slice)
            #arr = np.array(vols)
            #

            # Apply the transformations to the entire 3D image
        if self.transform:
            x_transformed = self.transform(x)
        else:
            x_transformed = x

        x_transformed = x_transformed.unsqueeze(0)
        # Assuming x_transformed has shape (H, W, D)
        # You can slice along the third dimension (D) to get individual slices
        #slices = [x_transformed[:, :, i] for i in range(x_transformed.shape[2])]
        slices = [x_transformed[:, :, i * 6 - 1] for i in range(1, 8)]
        # Convert the list of 2D slices to a 3D tensor
        x = torch.stack(slices)
        #print('klk co;osumadre', x.shape)
        #x = x.unsqueeze(1)


        return x, y

class CustomDataset2D(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if self.transform:
            x_pil = transforms.ToPILImage()(x)
            x_pill_trans = self.transform(x_pil)
            #x = x_pill_trans
            x = transforms.ToTensor()(x_pill_trans)

        #else:
            #vols = []
            #x_pil = transforms.ToPILImage()(x)
            #x_pill_trans = self.transform(x)
            # x = x_pill_trans
            #x = transforms.ToTensor()(x_pill_trans)
            #slice = slices(x)
            #vols.append(slice)
            #arr = np.array(vols)
            #

        return x, y

def train_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, path=None):
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    for epoch in tqdm(range(num_epochs), desc='Training Progress'):
        model.train()
        for batch, (image, label) in enumerate(train_loader):
            image = image.permute(0, 2, 1, 3, 4)
            #print(image.shape)
            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_X = val_X.permute(0, 2, 1, 3, 4)
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total

        if total > 0:
            val_accuracy = 100 * correct / total
        else:
            val_accuracy = 0.0

        # Print epoch-wise loss and accuracy
        print('------------------------------------------------------------------------------------------------------------------------------------------')
        print(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
        #torch.save(model.state_dict(), 'checkpoint_ResNet50_2D.pt')
    #model.load_state_dict(torch.save(f'checkpoint_ResNet50_2D.pt'))

    return model, loss, val_loss

def train_validate_model2D(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, path=None):
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    for epoch in tqdm(range(num_epochs), desc='Training Progress'):
        model.train()
        for batch, (image, label) in enumerate(train_loader):
            #print(image.shape)
            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_X, val_y in val_loader:
                val_outputs = model(val_X)
                val_loss += criterion(val_outputs, val_y).item()
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_y.size(0)
                correct += (predicted == val_y).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total

        if total > 0:
            val_accuracy = 100 * correct / total
        else:
            val_accuracy = 0.0

        # Print epoch-wise loss and accuracy
        print('------------------------------------------------------------------------------------------------------------------------------------------')
        print(f"Epoch [{epoch + 1}/{num_epochs}]: Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
        #torch.save(model.state_dict(), 'checkpoint_ResNet50_2D.pt')
    #model.load_state_dict(torch.save(f'checkpoint_ResNet50_2D.pt'))

    return model, loss, val_loss

def test_model2D(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the loss
            loss = criterion(output, target)
            # Update test loss
            test_loss += loss.item() * data.size(0)

            # Convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # Compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            # Calculate test accuracy for each object class
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

                # Collect predictions
                predictions.append(pred[i].item())

    # Calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(3):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    return predictions


def test_model3D(model, test_loader, criterion):
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.permute(0, 2, 1, 3, 4)
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the loss
            loss = criterion(output, target)
            # Update test loss
            #test_loss += loss.item() * data.size(0)

            # Convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # Compare predictions to true label
            #correct = np.squeeze(pred.eq(target.data.view_as(pred)))

            # Calculate test accuracy for each object class
            for i in range(len(target)):
                # Collect predictions
                predictions.append(pred[i].item())

    return predictions
