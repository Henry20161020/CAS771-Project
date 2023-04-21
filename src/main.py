from Dataset import FirstDataset
import torch
import matplotlib.pyplot as plt
import torchvision
import pandas as pd
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




def label_transform(df):
    return torch.tensor([int(a[1]) for a in df.iloc[:,0].str.split(' ')])

#
def normalize(nparr):
    return (nparr - nparr.min()) / (nparr.max() - nparr.min())

def image_transform():
    image = normalize(np.load("../data1/train/0.npy"))
    print(image.shape, image.dtype)
    #plt.imsave('test.png', image, cmap='binary')
    print(image)
    print(image.max(), image.min())



class MyModel(nn.Module):
    def __init__(self, key_size, channel):
        super(MyModel, self).__init__()
        self.key_size = key_size
        self.channel = channel
        self.kblock_size = key_size ** 2 * channel
        self.linear = nn.Linear(in_features=self.kblock_size, 
                                out_features=self.kblock_size, bias=False)
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)
        self.mapping = self.construct_mapping(4,32,3)
    
    def construct_mapping(self,key_size, width, channel):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(key_size * channel * width)
        #print(s.view(1,112))
        #print(source.view(4,28)) 
        # Reshape the tensor to [4, 28] with the desired pattern
        x = s.clone().reshape(key_size, width, channel)
        x_blocks = [x[i:i+key_size, j:j+key_size, :].contiguous() \
            for i in range(0, key_size, key_size) for j in range(0, width, key_size)]
        #print(x_blocks[0])
        x_transformed = [torch.reshape(block, (key_size ** 2 * 3, )) for block in x_blocks]
        #print(x_transformed[0])
        x_stacked = torch.stack(x_transformed, dim=0)
        #print(x_stacked[0])
        x_reshaped = x_stacked.reshape( key_size, -1)
        #print(x_reshaped[0], x_reshaped.shape)
        x_split = torch.split(x_reshaped, 1, dim=0)
        #print(x_split[0], x_split[0].shape)
        t = torch.cat(x_split, dim=0)
        #print(x_cat[0], x_cat.shape)

        #print(t.view(4,28))
        return torch.cat((s.reshape(1,key_size * channel * width),
                          t.reshape(1,key_size * channel * width)), dim=0)

    def tensor_map(self, input_t, mapping):
        output_t = torch.zeros_like(input_t)
        for i in range(mapping.shape[1]):
            output_t[:,mapping[1, i]] = input_t[:,mapping[0,i]]
        return output_t

    def decrypt(self, x, key):
        key_size = self.key_size
        #print(x[0][0:4])
        #x = x.to(device)
        x_blocks = [x[:, i:i+key_size, j:j+key_size, :].contiguous() \
            for i in range(0, x.size(dim=1), key_size) for j in range(0, x.size(dim=2), key_size)]
        #print("the first block:", x_blocks[0][0])
        #print(len(x_blocks))
        decrypt_matrix = nn.Linear(self.kblock_size,self.kblock_size, bias=False).to(device)
        #print(decrypt_matrix.weight.shape)
        with torch.no_grad():
            decrypt_matrix.weight.fill_(0)
            decrypt_matrix.weight.add_(key)
        x_transformed = [decrypt_matrix(block.reshape(x.shape[0], key_size ** 2 * 3)) \
                for block in x_blocks]
        #print("transformed: ", x_transformed[0].shape)
        x_stacked = torch.stack(x_transformed, dim=1)
        #print("stacked: ", x_stacked.shape)
        #print(x_stacked[0][0].shape)
        x_reshaped = x_stacked.reshape(x.shape[0], (x.size(dim=2) // key_size), -1)
        #print("reshaped: ", x_reshaped.shape)
        x_split = torch.split(x_reshaped, 1, dim=1)
        #print("split: ", len(x_split), x_split[0].shape)
        x_mapped = [self.tensor_map(t.reshape(x.shape[0], key_size * x.size(dim=2) * 3),\
            self.mapping) for t in x_split]
        #print(x_mapped[0][0])
        x_cat = torch.cat(x_mapped, dim=1)
        x_final = x_cat.reshape(x.shape[0], x.size(dim=2), x.size(dim=2), 3)
        #print((x == x_final)[0][0:4])
        return x_final
    
    def forward(self, x):
        key_size = self.key_size
        #print(x[0][0:4])
        x_blocks = [x[:, i:i+key_size, j:j+key_size, :].contiguous() \
            for i in range(0, x.size(dim=1), key_size) for j in range(0, x.size(dim=2), key_size)]
        #print("the first block:", x_blocks[0][0])
        #print(len(x_blocks))
        #print(x_blocks[0].shape, x.shape[0])
        x_transformed = [self.linear(block.view(x.shape[0], \
                       key_size ** 2 * 3)) for block in x_blocks]
        #print("transformed: ", x_transformed[0].shape)
        x_stacked = torch.stack(x_transformed, dim=1)
        #print("stacked: ", x_stacked.shape)
        #print(x_stacked[0][0].shape)
        x_reshaped = x_stacked.reshape(x.shape[0], (x.size(dim=2) // key_size), -1)
        #print("reshaped: ", x_reshaped.shape)
        x_split = torch.split(x_reshaped, 1, dim=1)
        #print("split: ", len(x_split), x_split[0].shape)
        x_mapped = [self.tensor_map(t.reshape(x.shape[0], key_size * x.size(dim=2) * 3),\
            self.mapping) for t in x_split]
        #print(x_mapped[0][0])
        x_cat = torch.cat(x_mapped, dim=1)
        x_final = x_cat.reshape(x.shape[0], x.size(dim=2), x.size(dim=2), 3)
        x_final = x_final.permute(0,3,1,2)
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv1(x_final), 2))
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x_final)), 2))
        print(x_final.shape)
        x_final = x_final.reshape(-1, 500)
        #print(x_final.shape)
        x_final = F.relu(self.fc1(x_final))
        #print(x_final.shape)
        x_final = F.dropout(x_final, training=self.training)
        #print(x_final.shape)
        x_final = self.fc2(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)
    
class CifarModel(nn.Module):
    def __init__(self, key_size, channel):
        super(CifarModel, self).__init__()
        self.key_size = key_size
        self.channel = channel
        self.kblock_size = key_size ** 2 * channel
        self.linear = nn.Linear(in_features=self.kblock_size, 
                                out_features=self.kblock_size, bias=False)
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc = nn.Linear(4*4*128, 10)
        self.mapping = self.construct_mapping(4,32,3)
    
    def construct_mapping(self,key_size, width, channel):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(key_size * channel * width)
        #print(s.view(1,112))
        #print(source.view(4,28)) 
        # Reshape the tensor to [4, 28] with the desired pattern
        x = s.clone().reshape(key_size, width, channel)
        x_blocks = [x[i:i+key_size, j:j+key_size, :].contiguous() \
            for i in range(0, key_size, key_size) for j in range(0, width, key_size)]
        #print(x_blocks[0])
        x_transformed = [torch.reshape(block, (key_size ** 2 * 3, )) for block in x_blocks]
        #print(x_transformed[0])
        x_stacked = torch.stack(x_transformed, dim=0)
        #print(x_stacked[0])
        x_reshaped = x_stacked.reshape( key_size, -1)
        #print(x_reshaped[0], x_reshaped.shape)
        x_split = torch.split(x_reshaped, 1, dim=0)
        #print(x_split[0], x_split[0].shape)
        t = torch.cat(x_split, dim=0)
        #print(x_cat[0], x_cat.shape)

        #print(t.view(4,28))
        return torch.cat((s.reshape(1,key_size * channel * width),
                          t.reshape(1,key_size * channel * width)), dim=0)

    def tensor_map(self, input_t, mapping):
        output_t = torch.zeros_like(input_t)
        for i in range(mapping.shape[1]):
            output_t[:,mapping[1, i]] = input_t[:,mapping[0,i]]
        return output_t

    def decrypt(self, x, key):
        key_size = self.key_size
        #print(x[0][0:4])
        #x = x.to(device)
        print(x.shape)
        #x = x.permute(0,2,3,1)
        x_blocks = [x[:, i:i+key_size, j:j+key_size, :].contiguous() \
            for i in range(0, x.size(dim=1), key_size) for j in range(0, x.size(dim=2), key_size)]
        print("the first block:", x_blocks[0].shape)
        print(len(x_blocks))
        decrypt_matrix = nn.Linear(self.kblock_size,self.kblock_size, bias=False).to(device)
        print(decrypt_matrix.weight.shape)
        with torch.no_grad():
            decrypt_matrix.weight.fill_(0)
            decrypt_matrix.weight.add_(key)
        x_transformed = [decrypt_matrix(block.reshape(x.shape[0], key_size ** 2 * 3)) \
                for block in x_blocks]
        #print("transformed: ", x_transformed[0].shape)
        x_stacked = torch.stack(x_transformed, dim=1)
        #print("stacked: ", x_stacked.shape)
        #print(x_stacked[0][0].shape)
        x_reshaped = x_stacked.reshape(x.shape[0], (x.size(dim=2) // key_size), -1)
        #print("reshaped: ", x_reshaped.shape)
        x_split = torch.split(x_reshaped, 1, dim=1)
        #print("split: ", len(x_split), x_split[0].shape)
        x_mapped = [self.tensor_map(t.reshape(x.shape[0], key_size * x.size(dim=2) * 3),\
            self.mapping) for t in x_split]
        #print(x_mapped[0][0])
        x_cat = torch.cat(x_mapped, dim=1)
        x_final = x_cat.reshape(x.shape[0], x.size(dim=2), x.size(dim=2), 3)
        #print((x == x_final)[0][0:4])
        return x_final
    
    def forward(self, x, stage):
        key_size = self.key_size
        if stage == 2:
            #print(x[0][0:4])
            #print(x.shape)
            x_blocks = [x[:, i:i+key_size, j:j+key_size, :].contiguous() \
                for i in range(0, x.size(dim=1), key_size) for j in range(0, x.size(dim=2), key_size)]
            #print("the first block:", x_blocks[0][0])
            #print(len(x_blocks))
            #print(x_blocks[0].shape, x.shape[0])
            x_transformed = [self.linear(block.view(x.shape[0], \
                        key_size ** 2 * 3)) for block in x_blocks]
            #print("transformed: ", x_transformed[0].shape)
            x_stacked = torch.stack(x_transformed, dim=1)
            #print("stacked: ", x_stacked.shape)
            #print(x_stacked[0][0].shape)
            x_reshaped = x_stacked.reshape(x.shape[0], (x.size(dim=2) // key_size), -1)
            #print("reshaped: ", x_reshaped.shape)
            x_split = torch.split(x_reshaped, 1, dim=1)
            #print("split: ", len(x_split), x_split[0].shape)
            x_mapped = [self.tensor_map(t.reshape(x.shape[0], key_size * x.size(dim=2) * 3),\
                self.mapping) for t in x_split]
            #print(x_mapped[0][0])
            x_cat = torch.cat(x_mapped, dim=1)
            x_final = x_cat.reshape(x.shape[0], x.size(dim=2), x.size(dim=2), 3)
            x_final = x_final.permute(0,3,1,2)
        else:
           x_final = x
        #print(x_final.shape)
        x_final = self.maxpool(F.relu(self.conv1(x_final)))
        #print(x_final.shape)
        x_final = self.maxpool(F.relu(self.conv2(x_final)))
        #print(x_final.shape)
        x_final = self.maxpool(F.relu(self.conv3(x_final)))
        #print(x_final.shape)
        #x_final = nn.Flatten(x_final)
        x_final = x_final.reshape(-1, 2048)
        #print(x_final.shape)
        x_final = self.fc(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)

#print(construct_mapping(4,32,3))
#print(torch.equal(network.decrypt(example_data, torch.eye(48)),example_data))
#print(data.shape, data.max(), data.min())
#plt.imsave('test.png', normalize(dataset1[0][0]), cmap='binary', format='png')
def train(epoch, stage):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    data, target = data.to(device), target.to(device)
    output = network(data, stage)
    #print(output.shape, target.shape)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')

def test(stage):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = network(data, stage)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  #mse.append(torch.mean(diff(data, encrypt(data_enc, network.linear.weight)).squeeze(), dim=0).item())
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))



def decrypt_imgs(decryption_key, source):
    #print(checkpoint["conv2.weight"].shape)
    #decrypted_img = normalize(network.decrypt(example_data.to(device),
    #         decryption_key)[0]).cpu().detach().numpy()
    #plt.imsave('test.png', decrypted_img, cmap='binary')
    count = 0
    dict_imgs_dec={}
    dict_imgs_enc={}
    for i in range(10):
       dict_imgs_dec[i] = []
       dict_imgs_enc[i] = []
    #print(dict_imgs)
    if source is None:
        for batch_idx, (data, target) in enumerate(train_loader):
            decrypted_img = network.decrypt(data.to(device), decryption_key.to(device))
            for i in range(data.shape[0]):
                #print(target[i].item())
                print(decrypted_img[i].shape, data[i].shape)
                if len(dict_imgs_dec[target[i].item()]) < 10: 
                    dict_imgs_dec[target[i].item()].append(normalize(decrypted_img[i]).cpu().\
                                                    detach().numpy())
                    dict_imgs_enc[target[i].item()].append(normalize(data[i]).cpu().\
                                                    detach().numpy())
                    count += 1
                if count == 100:
                    return dict_imgs_dec, dict_imgs_enc
    else:
        dict_imgs_enc = source
        for i in range(1):
           for img in source[i]:
              print(type(img))
              #network.decrypt
              #dict_imgs_dec[i].append()
    return dict_imgs_dec, dict_imgs_enc

# gb_image is a numpy array
# return a numpy arrary, normalized in [0, 1]
def rgb_to_grayscale(rgb_image):
    return normalize(np.dot(rgb_image, [0.2989, 0.5870, 0.1140]))

# img is a numpy arrary ranging in [0, 1]
def contrast(img):
  # Apply the transform to the image
  #print(img)
  return np.where(img > 0.5, 1, 0)
  #result = torch.where(img_normalized > 0.5, torch.tensor([1.]), torch.tensor([0.]))
  #return result

# dict_imgs is a numpy arrary
def save_images(dict_imgs, path):
   for (label, img_list) in dict_imgs.items():
      count = 0
      for img in img_list:
        plt.imsave(path + '/img' + str(label) + str(count) + '.png', 
                   img, cmap='binary')
        count += 1


def draw_curve():
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('curve.png')


#df = pd.read_csv("../data1/test_label.txt", sep=' ', header=None)
n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#print(label_transform(df))
#image_transform()
#print(df)

dataset1 = FirstDataset("../data1/train_label.txt", "../data1/train/", 
                        transform=torchvision.transforms.Compose([
                               torch.from_numpy,
                               normalize]))
data = dataset1[0][0]
print(data.shape)
train_dataset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform = torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='data/', download=True, train=False, transform = torchvision.transforms.ToTensor())

#for CIFAR
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)


examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
""" 
print(example_data.shape, example_targets.shape)
print(example_targets[0]) """

network = CifarModel(4,3)
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
mse = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs*2 + 1)]
"""
test(1)
for epoch in range(1, n_epochs + 1):
  train(epoch, 1)
  test(1)

for name, param in network.named_parameters():
    if 'conv' in name or 'fc' in name:
        param.requires_grad = False

#for dataset 1
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size_test, shuffle=True)

for epoch in range(n_epochs + 1, 2 * n_epochs + 1):
  train(epoch, 2)
  test(2)


draw_curve() 

"""

#for dataset 1
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size_test, shuffle=True)

# Load the .pth file for debugging
network.to(device)
checkpoint = torch.load("model_bak.pth")

# Print the keys of the checkpoint dictionary
#print(checkpoint.keys())

# Access the model parameters
#print(checkpoint["linear.bias"])
decryption_key =checkpoint["linear.weight"]

#w_matrix = torch.rand(48, 48)
#w_matrix_m1 = torch.inverse(w_matrix)

#enc, ori = decrypt_imgs(w_matrix, None)
#dec, _ = decrypt_imgs(w_matrix_m1, enc)
dec, enc = decrypt_imgs(decryption_key, None)
save_images(dec, '../decrypted_images')
save_images(enc, '../encrypted_images')
#save_images(ori, '../original_images')


#torch.manual_seed(42)

#my_list = [45000,5000]

#train_ds, val_ds = torch.utils.data.random_split(train_dataset, my_list)

#configs = {'random_seed' : 42, 'val_size' : 5000, 'train_size' : 45000, 'pin_memory':True,'optimizer':'Adam','batch_size':64,'lr':0.001 }
#train_data = torch.utils.data.DataLoader(train_ds, configs['batch_size'], shuffle = True, pin_memory = configs['pin_memory'], num_workers = 2)
#val_data = torch.utils.data.DataLoader(val_ds, configs['batch_size'], shuffle = True, pin_memory = configs['pin_memory'], num_workers = 2)


