import torch
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

def normalize(tensor):
    return (tensor - torch.amin(tensor, dim=(2,3)).view(tensor.shape[0], 1, 1, 1)) / (torch.amax(tensor, dim=(2,3)) - torch.amin(tensor, dim=(2,3))).view(tensor.shape[0], 1, 1, 1) * 255

def diff(img1, img2):
  #print(img1.size(dim=2), img2.size(dim=3))
  return(torch.sum(torch.square(normalize(img1) - normalize(img2)), (2, 3)) / (img1.size(dim=2) * img2.size(dim=3)))

def encrypt(img, key):
  #fig, axs = plt.subplots(14, 7)
  assembled_image = torch.zeros_like(img)
  for i in range(7):
    for j in range(7):
      img_cropped = torchvision.transforms.functional.crop(img, top=4*i, left=4*j, height=4, width=4)
      #axs[i, j].imshow(img_cropped[0][0], interpolation='none')

      #print(img_cropped.shape)
      img_reshaped = img_cropped.contiguous().view(img.shape[0], 1, 16, 1)
      #print(img_reshaped[0][0])
      
      #print(w_matrix)
      #print(img_reshaped[0][0])
      #print(img_reshaped.shape)
      img_encrypted = torch.matmul(key.expand(img.shape[0], 1, 16, 16), img_reshaped)
      img_pieces = img_encrypted.view(img.shape[0], 1, 4, 4)
      #org = torch.matmul(torch.inverse(w_matrix), encrypted)
      #print(org)
      #axs[7+i, j].imshow(img_pieces[0][0], interpolation='none')
      assembled_image[:,:,4*i:4*i+4,4*j:4*j+4] = img_pieces
  return assembled_image

torch.manual_seed(1121)
        #print(img_reshaped.shape)
w_matrix = torch.rand(16, 16)
w_matrix_m1 = torch.inverse(w_matrix)


# MyModel is baseline. 
# Model2 removes max pooling.
# Model3 removes dropout.
# Model4 use kernel size 3 to reduct parameters.
class MyModel(nn.Module):
    def __init__(self, stage):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(in_features=16, out_features=16, bias=False)
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.stage = stage
    
    def construct_mapping(self):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(112)
        #print(s.view(1,112))
        #print(source.view(4,28))
        # Reshape the tensor to [4, 28] with the desired pattern
        t = s.clone().reshape(4,28)
        t[0] = torch.tensor([0, 1, 2, 3, 16,17,18,19, 32,33,34,35, 48,49,50,51, 64,65,66,67, 80,81,82,83, 96, 97, 98, 99])
        for i in range(1, 4):
            t[i] = t[0] + i*4
        #print(t.view(4,28))
        return torch.cat((t.reshape(1,112),s.reshape(1,112)), dim=0)

    def tensor_map(self, input_t, mapping):
      output_t = torch.zeros_like(input_t)
      for i in range(mapping.shape[1]):
        output_t[:,:,mapping[1, i]] = input_t[:,:,mapping[0,i]]
      return output_t 
      
    def forward(self, x):
        if self.stage == 2:
          # Split the input tensor into 4x4 blocks
          x_blocks = [x[:, :, i:i+4, j:j+4].contiguous() for i in range(0, 28, 4) for j in range(0, 28, 4)]
          # Apply the linear transformation to each block
          #print(x_blocks[0].shape)
          #print(x_blocks[0].shape, x.shape[0])
          x_transformed = [self.linear(block.view(x.shape[0], 1, 16)) for block in x_blocks]
          x_stacked = torch.stack(x_transformed, dim=2)
          # Concatenate the transformed groups along the feature dimension
          x_reshaped = x_stacked.reshape(x.shape[0], 1, 49, 16)
          x_split = torch.split(x_reshaped, 7, dim=2)
          x_mapped = [self.tensor_map(t.reshape(x.shape[0], 1, 112), self.mapping).reshape(x.shape[0],1,4, 28) for t in x_split]
          # Concatenate the tensors along the third dimension (feature dimension)
          x_cat = torch.cat(x_mapped, dim=2)
          # Reshape the concatenated tensor to have shape (batch_size, 1, 28, 28)
          x_final = x_cat.reshape(x.shape[0], 1, 28, 28)
        else:
          x_final = x
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv1(x_final), 2))
        print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x_final)), 2))
        #print(x_final.shape)
        x_final = x_final.view(-1, 320)
        #print(x_final.shape)
        x_final = F.relu(self.fc1(x_final))
        #print(x_final.shape)
        x_final = F.dropout(x_final, training=self.training)
        #print(x_final.shape)
        x_final = self.fc2(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)

class MyModel2(nn.Module):
    def __init__(self, stage):
        super(MyModel2, self).__init__()
        self.linear = nn.Linear(in_features=16, out_features=16, bias=False)
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8000, 50)
        self.fc2 = nn.Linear(50, 10)
        self.stage = stage
    
    def construct_mapping(self):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(112)
        #print(s.view(1,112))
        #print(source.view(4,28))
        # Reshape the tensor to [4, 28] with the desired pattern
        t = s.clone().reshape(4,28)
        t[0] = torch.tensor([0, 1, 2, 3, 16,17,18,19, 32,33,34,35, 48,49,50,51, 64,65,66,67, 80,81,82,83, 96, 97, 98, 99])
        for i in range(1, 4):
            t[i] = t[0] + i*4
        #print(t.view(4,28))
        return torch.cat((t.reshape(1,112),s.reshape(1,112)), dim=0)

    def tensor_map(self, input_t, mapping):
      output_t = torch.zeros_like(input_t)
      for i in range(mapping.shape[1]):
        output_t[:,:,mapping[1, i]] = input_t[:,:,mapping[0,i]]
      return output_t 
      
    def forward(self, x):
        if self.stage == 2:
          # Split the input tensor into 4x4 blocks
          x_blocks = [x[:, :, i:i+4, j:j+4].contiguous() for i in range(0, 28, 4) for j in range(0, 28, 4)]
          # Apply the linear transformation to each block
          #print(x_blocks[0].shape)
          #print(x_blocks[0].shape, x.shape[0])
          x_transformed = [self.linear(block.view(x.shape[0], 1, 16)) for block in x_blocks]
          x_stacked = torch.stack(x_transformed, dim=2)
          # Concatenate the transformed groups along the feature dimension
          x_reshaped = x_stacked.reshape(x.shape[0], 1, 49, 16)
          x_split = torch.split(x_reshaped, 7, dim=2)
          x_mapped = [self.tensor_map(t.reshape(x.shape[0], 1, 112), self.mapping).reshape(x.shape[0],1,4, 28) for t in x_split]
          # Concatenate the tensors along the third dimension (feature dimension)
          x_cat = torch.cat(x_mapped, dim=2)
          # Reshape the concatenated tensor to have shape (batch_size, 1, 28, 28)
          x_final = x_cat.reshape(x.shape[0], 1, 28, 28)
        else:
          x_final = x
        #print(x_final.shape)
        x_final = F.relu(self.conv1(x_final))
        #print(x_final.shape)
        x_final = F.relu(self.conv2_drop(self.conv2(x_final)))
        #print(x_final.shape)
        x_final = x_final.view(-1, 8000)
        #print(x_final.shape)
        x_final = F.relu(self.fc1(x_final))
        #print(x_final.shape)
        x_final = F.dropout(x_final, training=self.training)
        #print(x_final.shape)
        x_final = self.fc2(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)

class MyModel3(nn.Module):
    def __init__(self, stage):
        super(MyModel3, self).__init__()
        self.linear = nn.Linear(in_features=16, out_features=16, bias=False)
        self.conv = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.stage = stage
    
    def construct_mapping(self):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(112)
        #print(s.view(1,112))
        #print(source.view(4,28))
        # Reshape the tensor to [4, 28] with the desired pattern
        t = s.clone().reshape(4,28)
        t[0] = torch.tensor([0, 1, 2, 3, 16,17,18,19, 32,33,34,35, 48,49,50,51, 64,65,66,67, 80,81,82,83, 96, 97, 98, 99])
        for i in range(1, 4):
            t[i] = t[0] + i*4
        #print(t.view(4,28))
        return torch.cat((t.reshape(1,112),s.reshape(1,112)), dim=0)

    def tensor_map(self, input_t, mapping):
      output_t = torch.zeros_like(input_t)
      for i in range(mapping.shape[1]):
        output_t[:,:,mapping[1, i]] = input_t[:,:,mapping[0,i]]
      return output_t 
      
    def forward(self, x):
        if self.stage == 2:
          # Split the input tensor into 4x4 blocks
          x_blocks = [x[:, :, i:i+4, j:j+4].contiguous() for i in range(0, 28, 4) for j in range(0, 28, 4)]
          # Apply the linear transformation to each block
          #print(x_blocks[0].shape)
          #print(x_blocks[0].shape, x.shape[0])
          x_transformed = [self.linear(block.view(x.shape[0], 1, 16)) for block in x_blocks]
          x_stacked = torch.stack(x_transformed, dim=2)
          # Concatenate the transformed groups along the feature dimension
          x_reshaped = x_stacked.reshape(x.shape[0], 1, 49, 16)
          x_split = torch.split(x_reshaped, 7, dim=2)
          x_mapped = [self.tensor_map(t.reshape(x.shape[0], 1, 112), self.mapping).reshape(x.shape[0],1,4, 28) for t in x_split]
          # Concatenate the tensors along the third dimension (feature dimension)
          x_cat = torch.cat(x_mapped, dim=2)
          # Reshape the concatenated tensor to have shape (batch_size, 1, 28, 28)
          x_final = x_cat.reshape(x.shape[0], 1, 28, 28)
        else:
          x_final = x
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv1(x_final), 2))
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv2(x_final), 2))
        #print(x_final.shape)
        x_final = x_final.view(-1, 320)
        #print(x_final.shape)
        x_final = F.relu(self.fc1(x_final))
        #print(x_final.shape)
        x_final = F.dropout(x_final, training=self.training)
        #print(x_final.shape)
        x_final = self.fc2(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)
    

class MyModel4(nn.Module):
    def __init__(self, stage):
        super(MyModel4, self).__init__()
        self.linear = nn.Linear(in_features=16, out_features=16, bias=False)
        self.conv = nn.Conv2d(1, 10, kernel_size=3)
        self.conv3 = nn.Conv2d(10, 20, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)
        self.stage = stage
    
    def construct_mapping(self):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(112)
        #print(s.view(1,112))
        #print(source.view(4,28))
        # Reshape the tensor to [4, 28] with the desired pattern
        t = s.clone().reshape(4,28)
        t[0] = torch.tensor([0, 1, 2, 3, 16,17,18,19, 32,33,34,35, 48,49,50,51, 64,65,66,67, 80,81,82,83, 96, 97, 98, 99])
        for i in range(1, 4):
            t[i] = t[0] + i*4
        #print(t.view(4,28))
        return torch.cat((t.reshape(1,112),s.reshape(1,112)), dim=0)

    def tensor_map(self, input_t, mapping):
      output_t = torch.zeros_like(input_t)
      for i in range(mapping.shape[1]):
        output_t[:,:,mapping[1, i]] = input_t[:,:,mapping[0,i]]
      return output_t 
      
    def forward(self, x):
        if self.stage == 2:
          # Split the input tensor into 4x4 blocks
          x_blocks = [x[:, :, i:i+4, j:j+4].contiguous() for i in range(0, 28, 4) for j in range(0, 28, 4)]
          # Apply the linear transformation to each block
          #print(x_blocks[0].shape)
          #print(x_blocks[0].shape, x.shape[0])
          x_transformed = [self.linear(block.view(x.shape[0], 1, 16)) for block in x_blocks]
          x_stacked = torch.stack(x_transformed, dim=2)
          # Concatenate the transformed groups along the feature dimension
          x_reshaped = x_stacked.reshape(x.shape[0], 1, 49, 16)
          x_split = torch.split(x_reshaped, 7, dim=2)
          x_mapped = [self.tensor_map(t.reshape(x.shape[0], 1, 112), self.mapping).reshape(x.shape[0],1,4, 28) for t in x_split]
          # Concatenate the tensors along the third dimension (feature dimension)
          x_cat = torch.cat(x_mapped, dim=2)
          # Reshape the concatenated tensor to have shape (batch_size, 1, 28, 28)
          x_final = x_cat.reshape(x.shape[0], 1, 28, 28)
        else:
          x_final = x
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv(x_final), 2))
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x_final)), 2))
        #print(x_final.shape)
        x_final = x_final.view(-1, 500)
        #print(x_final.shape)
        x_final = F.relu(self.fc1(x_final))
        #print(x_final.shape)
        x_final = F.dropout(x_final, training=self.training)
        #print(x_final.shape)
        x_final = self.fc2(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)
    
class MyModel5(nn.Module):
    def __init__(self, stage):
        super(MyModel5, self).__init__()
        self.linear = nn.Linear(in_features=16, out_features=16, bias=False)
        self.conv = nn.Conv2d(1, 5, kernel_size=3)
        self.conv3 = nn.Conv2d(5, 10, kernel_size=3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mapping = self.construct_mapping()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)
        self.fc3 = nn.Linear(250, 10)
        self.stage = stage
    
    def construct_mapping(self):
        # Create a tensor with shape [112] and values [0, 1, 2, ..., 111]
        s = torch.arange(112)
        #print(s.view(1,112))
        #print(source.view(4,28))
        # Reshape the tensor to [4, 28] with the desired pattern
        t = s.clone().reshape(4,28)
        t[0] = torch.tensor([0, 1, 2, 3, 16,17,18,19, 32,33,34,35, 48,49,50,51, 64,65,66,67, 80,81,82,83, 96, 97, 98, 99])
        for i in range(1, 4):
            t[i] = t[0] + i*4
        #print(t.view(4,28))
        return torch.cat((t.reshape(1,112),s.reshape(1,112)), dim=0)

    def tensor_map(self, input_t, mapping):
      output_t = torch.zeros_like(input_t)
      for i in range(mapping.shape[1]):
        output_t[:,:,mapping[1, i]] = input_t[:,:,mapping[0,i]]
      return output_t 
      
    def forward(self, x):
        if self.stage == 2:
          # Split the input tensor into 4x4 blocks
          x_blocks = [x[:, :, i:i+4, j:j+4].contiguous() for i in range(0, 28, 4) for j in range(0, 28, 4)]
          # Apply the linear transformation to each block
          #print(x_blocks[0].shape)
          #print(x_blocks[0].shape, x.shape[0])
          x_transformed = [self.linear(block.view(x.shape[0], 1, 16)) for block in x_blocks]
          x_stacked = torch.stack(x_transformed, dim=2)
          # Concatenate the transformed groups along the feature dimension
          x_reshaped = x_stacked.reshape(x.shape[0], 1, 49, 16)
          x_split = torch.split(x_reshaped, 7, dim=2)
          x_mapped = [self.tensor_map(t.reshape(x.shape[0], 1, 112), self.mapping).reshape(x.shape[0],1,4, 28) for t in x_split]
          # Concatenate the tensors along the third dimension (feature dimension)
          x_cat = torch.cat(x_mapped, dim=2)
          # Reshape the concatenated tensor to have shape (batch_size, 1, 28, 28)
          x_final = x_cat.reshape(x.shape[0], 1, 28, 28)
        else:
          x_final = x
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv(x_final), 2))
        #print(x_final.shape)
        x_final = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x_final)), 2))
        #print(x_final.shape)
        x_final = x_final.view(-1, 250)
        #print(x_final.shape)
        x_final = F.relu(self.fc3(x_final))
        #print(x_final.shape)
        x_final = F.dropout(x_final, training=self.training)
        #print(x_final.shape)
        #x_final = self.fc2(x_final)
        #print(x_final.shape)
        return F.log_softmax(x_final)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = MyModel(1)
network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum) 

train_losses = []
train_counter = []
test_losses = []
train_mse = []
test_mse = []
train_mse_ratio = []
test_mse_ratio = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch, stage):
  network.train()
  mse_dec = torch.empty(0).to(device)
  mse_enc = torch.empty(0).to(device)
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    if stage == 2:
      data_enc = encrypt(data, w_matrix).to(device)
    target = target.to(device)
    output = network(data_enc)
    #print(output.shape, target.shape)
    loss = F.nll_loss(output, target)
    diff_dec = diff(data.to(device), encrypt(data_enc, network.linear.weight)).squeeze()
    diff_enc = diff(data.to(device), data_enc).squeeze()
    mse_dec = torch.cat((mse_dec, diff_dec), dim=0)
    mse_enc = torch.cat((mse_enc, diff_enc), dim=0)
    #print(diff_enc[:10], diff_dec[:10])
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_mse.append(torch.mean(mse_dec, dim=0).item())
      train_mse_ratio.append(torch.median(torch.div(mse_enc, mse_dec)).item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'model.pth')
      torch.save(optimizer.state_dict(), 'optimizer.pth')

def test(stage):
  network.eval()
  test_loss = 0
  correct = 0
  mse_dec = torch.empty(0).to(device)
  mse_enc = torch.empty(0).to(device)
  with torch.no_grad():
    for data, target in test_loader:
      if stage == 2:
        data_enc = encrypt(data, w_matrix).to(device)
      else:
        data_enc = 
      target = target.to(device)
      output = network(data_enc)
      #print(output.shape, target.shape)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      diff_dec = diff(data.to(device), encrypt(data_enc, network.linear.weight)).squeeze()
      diff_enc = diff(data.to(device), data_enc).squeeze()
      mse_dec = torch.cat((mse_dec, diff_dec), dim=0)
      mse_enc = torch.cat((mse_enc, diff_enc), dim=0)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  test_mse.append(torch.mean(mse_dec, dim=0).item())
  test_mse_ratio.append(torch.median(torch.div(mse_enc, mse_dec)).item())
  #for i in range(10):
  #  print(diff(data, encrypt(data_enc, network.linear.weight)).squeeze()[i].item())
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test(1)
for epoch in range(1, n_epochs + 1):
  train(epoch, 1)
  test(1)

for name, para in network.named_parameters():
   if 'conv' in name or 'fc' in name:
      para.requires_grad = False

test(2)
for epoch in range(1, n_epochs + 1):
  train(epoch, 2)
  test(2)

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
#print(test_counter, test_losses)
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig('images/loss.png')

fig = plt.figure()
plt.plot(train_counter, train_mse, color='blue')
plt.scatter(test_counter, test_mse, color='red')
plt.legend(['Train mse', 'Test mse'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('mse')
plt.savefig('images/mse.png')

fig = plt.figure()
plt.plot(train_counter, train_mse_ratio, color='blue')
plt.scatter(test_counter, test_mse_ratio, color='red')
plt.legend(['Train mse ratio', 'Test mse ratio'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('mse_ratio')
plt.savefig('images/mse_ratio.png')

# Load the .pth file for debugging
checkpoint = torch.load("model.pth")

# Print the keys of the checkpoint dictionary
print(checkpoint.keys())

# Access the model parameters
#print(checkpoint["linear.bias"])
decryption_key =checkpoint["linear.weight"].cpu().detach()
#print(decryption_key)
#print(checkpoint["conv2.weight"].shape)

def contrast(img):
  img_normalized = normalize(img)
  # Apply the transform to the image
  #print(img)
  result = torch.where(img_normalized > 127, torch.tensor([255.]), torch.tensor([0.]))
  return result



example_data = normalize(example_data)
img_encrypted = normalize(encrypt(example_data, w_matrix))
img_decrypted = normalize(encrypt(img_encrypted, decryption_key))
img_contrast = normalize(contrast(img_decrypted))


i = 4

# Ask the user for the lengths of the sides
a = int(diff(example_data, img_decrypted)[i][0].item())
b = int(diff(example_data, img_encrypted)[i][0].item())
c = int(diff(example_data, img_contrast)[i][0].item())
d = int(diff(example_data, example_data)[i][0].item())


fig, axs = plt.subplots(1, 4, figsize=(8,8))
axs[0].imshow(example_data[i][0], cmap = "gray", interpolation='none')
axs[0].set_title('Original.\n Diff : {}'.format(d))
axs[1].imshow(img_encrypted[i][0], cmap = "gray", interpolation='none')
axs[1].set_title('Encrypted. \nDiff : {}'.format(b))
axs[2].imshow(img_decrypted[i][0], cmap = "gray", interpolation='none')
axs[2].set_title('Decrypted. \nDiff : {}'.format(a))
axs[3].imshow(img_contrast[i][0], cmap = "gray", interpolation='none')
axs[3].set_title('Contrasted. \nDiff : {}'.format(c))
plt.savefig('images/comparison.png')