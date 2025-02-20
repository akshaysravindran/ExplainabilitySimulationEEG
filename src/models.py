
"""
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    Please contact Akshay Sujatha Ravindran for more info about licensing asujatharavindran@uh.edu
    via github issues section.
    **********************************************************************************
    Author:     Akshay Sujatha Ravindran
    Date:       04/20/21    
    **********************************************************************************
"""
import torch
import torch.nn as nn





class DeepNet(nn.Module):
    """
    Model to classify frequency resolution capability of CNN    
    
    """
    def __init__(self, num_units=64, conv_size=10, eeg_chan_num=62,pool_dim=(1, 2), strides_pool=(1, 2),num_classes=4):
        super(DeepNet, self).__init__()
        self.conv1 = nn.Conv2d(1, num_units, (1,conv_size))
        self.pool1 = nn.MaxPool2d(pool_dim, strides_pool)
        
        
        self.conv2 = nn.Conv2d(num_units, num_units, (1,conv_size))
        self.pool2 = nn.MaxPool2d(pool_dim, strides_pool)
        
        self.conv3 = nn.Conv2d(num_units, num_units, (1,conv_size))
        self.pool3 = nn.MaxPool2d(pool_dim, strides_pool)        
        
        self.conv4 = nn.Conv2d(num_units, num_units, (1,conv_size))
        self.pool4 = nn.MaxPool2d(pool_dim, strides_pool)
        
        # self.conv4 = nn.Conv2d(num_units, num_units, (1,conv_size))  
        # self.pool4 = nn.MaxPool2d(pool_dim, strides_pool)  
        
        self.conv5 = nn.Conv2d(num_units, num_units, (1,conv_size))  
        self.conv6 = nn.Conv2d(num_units, num_units, (eeg_chan_num,1))    

        
        self.fc1 = nn.Linear(224, num_units)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(num_units, num_classes)
        
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()
        self.relu7 = nn.ReLU()
    def forward(self, x):       
        
        x = self.pool2(self.relu2(self.conv2(self.pool1(self.relu1(self.conv1(x))))))
        x = self.pool4(self.relu4(self.conv4(self.pool3(self.relu3(self.conv3(x))))))
        x = self.relu6(self.conv6(self.relu5(self.conv5(x))))
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.dropout(self.relu7(self.fc1(x)))        
        
        x = self.fc2(x)
        return x
    
    
        
def train_model(model, train_loader, optimizer, criterion,device,reg_function1=None, reg_function2=None):
  """
  Function to train the  model

  Args:
    model : model to be trained     
    train_loader: data loader to get data to train the model on
    optimizer: type of optimizer to optimize the learning procedure
    criterion: Loss with which to evaluate the model on
    reg_function 1 and 2: Regularization functions if needed
  Returns:
    trained model
  """   
  print('Training')
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    if reg_function1 is None:
      loss = criterion(output, target)
    elif reg_function2 is None:
      loss = criterion(output, target)+ 0.001*reg_function1(model)
    else:
      loss = criterion(output, target) + 0.001*reg_function1(model) + 0.001*reg_function2(model)
    loss.backward()
    optimizer.step()

  return model




def validate_model(model, test_loader, criterion,device):
  """
  Evaluate the performance of the  Model

  Args:
    model : model being evaluated
    test_loader: data loader to get data to test the model on
    criterion: Loss with which to evaluate the model on

  Returns:
    test loss and accuracy
  """    

  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += criterion(output, target).item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  return test_loss, 100. * correct / len(test_loader.dataset)





def l1_reg(model):
  """
    Inputs: Pytorch model
    This function calculates the l1 norm of the all the tensors in the model
  """
  l1 = 0.0

  for param in model.parameters():
    l1 += torch.sum(torch.abs(param))

  return l1


def l2_reg(model):

  """
    Inputs: Pytorch model
    This function calculates the l2 norm of the all the tensors in the model
  """

  l2 = 0.0
  for param in model.parameters():
    l2 += torch.sum(torch.abs(param)**2)

  return l2
