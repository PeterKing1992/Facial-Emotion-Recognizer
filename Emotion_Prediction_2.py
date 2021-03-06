"""
Created on Tue Feb 20 2021

@author: Zhao Ji Wang
"""


import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def conv_block(in_channels, out_channels, pool = False): 
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size = 7, padding = 3), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True)]
    if pool: 
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers) 

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

class FacialEmotionConvolutionalModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)  #64 x 24 x 24
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128)) # 128 x 24 x 24
         
        self.conv3 = conv_block(128, 256, pool=True) #256 x 12 x 12
        self.conv4 = conv_block(256, 512, pool=True) #512 x 6 x 6
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512)) #512 x 6 x 6
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        
        out = self.classifier(out)
        return out

def fit(epochs, lr, model, train_loader, val_loader, weight_decay = 0, grad_clip = None, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr, weight_decay = weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        model.train() 
        
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    yb = F.softmax(yb, dim = 1)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

def load_in_image(): 
    #Load in own test image
    img_path="Face.png"
    image = Image.open(img_path)
    image = image.convert("L")
    image = ToTensor()(image) # unsqueeze to add artificial first dimension
    transf = transforms.Compose([transforms.Normalize(*stats)])
    image = transf(image)
    print("Image Shape: ", image.shape)
    return image

def test_own_image(model): 
    image = load_in_image()
    print("Predicted: ", emotions[predict_image(image, model)]) 

def save_model(model, filepath): 
    torch.save(model.state_dict(), filepath)
    
def load_model(model, filepath): 
    model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    
def show_image(img, label):
    print('Label: ', val_ds.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0), cmap = 'gray')
    
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

batch_size = 128
in_channels = 1 
num_classes = 7 
lr = 0.01
grad_clip = 0.05
weight_decay = 1e-4

file_dir = './fer2013' 
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

stats = ((0.5), (0.5))
GrayScaleTensorTransform  = transforms.Compose([transforms.Grayscale(), 
                                                transforms.RandomCrop(48, padding = 6, padding_mode = 'reflect'),
                                                transforms.RandomHorizontalFlip(), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize(*stats, inplace = True)
                                                ])

ValTransform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(*stats, inplace = True)
                                  ])

train_ds = ImageFolder(file_dir + '/train', transform = GrayScaleTensorTransform)

val_ds = ImageFolder(file_dir + '/test', transform = ValTransform)

device = get_default_device()


train_ldr = DataLoader(train_ds, batch_size, shuffle=True)
val_ldr = DataLoader(val_ds, batch_size)

train_loader = DeviceDataLoader(train_ldr, device)
val_loader = DeviceDataLoader(val_ldr, device)


# Logistic regression model
model1 = to_device(FacialEmotionConvolutionalModel(), device)

load_model(model1, './fer2013-convolutional2.pth') 

# history1 = fit(30, lr, model1, train_loader, val_loader, grad_clip = grad_clip) #, weight_decay = weight_decay) # Uncommment this line if you wish to train this model

test_own_image(model1) # Comment out this line if you wish to train this model. 

save_model(model1, 'fer2013-convolutional2.pth')

