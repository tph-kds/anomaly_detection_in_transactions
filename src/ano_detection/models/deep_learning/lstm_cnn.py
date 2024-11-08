import os
import sys

from .base_model import Base_DeepLearningModel
from src.ano_detection.logger import logger
from src.ano_detection.exception import MyException

import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F

class LSTM_CNN_Model(nn.Module, Base_DeepLearningModel):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 num_classes: int,
                 dropout_prob: float, 
                 kernel_size: int,
                 padding_size: int,
                 stride_size: int,  
                 eps: float, # 1e-05
                 momentum: float, # 0.1
                 offine: bool, # True
                 track_running_stats: bool, # True
                 device: str = 'cpu'):
        
        super(LSTM_CNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding_size)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding_size)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=padding_size)
        # self.fc = nn.Linear(hidden_size, num_classes)
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride_size, padding=padding_size)
        # self.timed = nn.TimeDistributed(self.fc)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norm = nn.BatchNorm1d(hidden_size, 
                                         eps=eps, 
                                         momentum=momentum, 
                                         affine=offine, 
                                         track_running_stats=track_running_stats
                                         )
        self.sigmoid = nn.Sigmoid()

        self.device = device

    def forward(self, x):
        """


        Args:
            x (torch.Tensor): input tensor of shape (n_samples, channels, n_features)

        Returns:    

            
        """
        # Set initial hidden and cell states
        hidden, cell = self.init_hidden(x.shape[0])
        # print(f"Hiden: {hidden.shape}") # (2, 32, 64)
        # print(f"Cell: {cell.shape}") # (2, 32, 64)


        # Apply CNN layers 
        out = self.CnnBlock(x) # (32, 4096)
        ## Platten 
        out = self.flatten(out) # (32, 4096)

        out = self.dropout(out) # (32, 4096)
        out = out.view(out.shape[0], self.lstm.hidden_size, self.lstm.hidden_size) # (32, 64, 64)

        # Forward propagate LSTM
        out = self.LstmBlock(out, hidden, cell)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        

        return out
    
    def init_hidden(self, batch_size):
        h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(self.device)
        return h, c
    
    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            h0 = torch.zeros(self.lstm.num_layers, self.input_size, self.lstm.hidden_size).to(self.device)
            c0 = torch.zeros(self.lstm.num_layers, self.input_size, self.lstm.hidden_size).to(self.device)
            out, _ = self.lstm(x, (h0, c0))
            out = out.reshape(out.shape[0], -1)
            out = self.relu(self.conv1(out))
            out = self.relu(self.conv2(out))
            out = self.fc(out)
            out = torch.sigmoid(out)
            return out
    
    def LstmBlock(self, x, hidden, cell):
        # Forward propagate LSTM
        out, _ = self.lstm(x, (hidden, cell))

        # Reshape output to (batch_size*seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)  # (32, 4096)
        # print(out.shape)
        
        # Apply fully connected layer
        out = self.FullyConnectedBlock(out, hidden_size=out.shape[1], num_classes=out.shape[1]*2) # out: tensor of shape (batch_size, hidden_size)
        out = self.FullyConnectedBlock(out, hidden_size=out.shape[1], num_classes=self.num_classes) # out: tensor of shape (batch_size, num_classes)
        out = self.sigmoid(out)
        return out

    def CnnBlock(self, x):
        """

        Args:
            x (torch.Tensor): input tensor of shape (n_samples, channels, n_features)

        Returns:

        """
        # Apply convolution layers 
        out = self.relu(self.conv1(x)) # out: tensor of shape (batch_size, channels, hidden_size)
        out = self.batch_norm(out) # out: tensor of shape (batch_size, channels, hidden_size)
        out = self.relu(self.conv2(out))
        out = self.batch_norm(out)
        out = self.relu(self.conv3(out))
        out = self.batch_norm(out)

        # Apply max pooling
        out = self.max_pool(out)

        out = self.dropout(out) # out: tensor of shape (batch_size, channels, n_features)

        out = out.reshape(out.shape[0], -1) # out: tensor of shape (batch_size, channels*n_features)
        # Apply fully connected layer
        out = self.FullyConnectedBlock(out, hidden_size=out.shape[1], num_classes=out.shape[1] * 2) # out: tensor of shape (batch_size, channels*n_features * 2)
        out = self.FullyConnectedBlock(out, hidden_size=out.shape[1], num_classes=out.shape[1]) # out: tensor of shape (batch_size, channels*n_features)
        out = self.FullyConnectedBlock(out, hidden_size=out.shape[1], num_classes=self.hidden_size*self.hidden_size) # out: tensor of shape (batch_size, hidden_size)
        # out = self.FullyConnectedBlock(out, hidden_size=out.shape[1], num_classes=self.hidden_size) # out: tensor of shape (batch_size, hidden_size)
        
        # print(out.shape)
        return out
    
    def FullyConnectedBlock(self, x, hidden_size=64, num_classes=1):
        self.fc = nn.Linear(hidden_size, num_classes)
        out = self.fc(x)
        return out
    


    def train_model(self, model, train_loader, optimizer, loss_function, num_epoch):
        model.train(True)
        print(f"Training model for {num_epoch} epochs")
        loss_train = 0.0

        for batch_index, batch in enumerate(train_loader):
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device).type(torch.long)

            # convert shape from (batch_size, n_features) to (batch_size, in_channels, n_features)
            data = data.reshape(data.shape[0], 1, data.shape[1])
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            # print(outputs.requires_grad)  # Should be True
            # print(labels.requires_grad)  # Should be False, as it's not a parameter
            # logits = torch.argmax(outputs, dim=1).type(torch.float32)
            loss = loss_function(outputs, labels)
            loss_train += loss.item()
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            if (batch_index) % 100 == 99: # Print every 100 mini-batches
                avg_loss_train = loss_train / 100
                print ('Step/Batch [{}/{}], Loss: {:.4f}'
                    .format(batch_index+1, len(train_loader), avg_loss_train))
                loss_train = 0.0

        print("****"*10)
        print()


    def validation_model(self, model, val_loader, loss_function, num_epoch):
        model.eval()
        model.train(False)
        print(f"Validating model for {num_epoch} epochs")
        los_val = 0.0

        for batch_index, batch in enumerate(val_loader):
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device).type(torch.long)

            # convert shape from (batch_size, n_features) to (batch_size, in_channels, n_features)
            data = data.reshape(data.shape[0], 1, data.shape[1])
            
            with torch.no_grad():
                # Forward pass
                outputs = model(data)
                loss = loss_function(outputs, labels)
                los_val += loss.item()
            
        avg_loss_val = los_val / len(val_loader)
        print ('Loss: {:.4f}'
            .format(avg_loss_val))
        print("****"*10)
        print()

    def test_model(self, model, test_loader, loss_function):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_index, batch in enumerate(test_loader):
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device).type(torch.long)

            # Reshape data if needed (e.g., for CNN+LSTM input format)
            data = data.reshape(data.shape[0], 1, data.shape[1])
            
            with torch.no_grad():
                outputs = model(data)
                loss = loss_function(outputs, labels)
                total_loss += loss.item()

                # Assuming binary or multi-class classification
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
            
        avg_loss = total_loss / len(test_loader)
        accuracy = total_correct / total_samples * 100

        print("Testing Results: ")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("****" * 10)
    

    
