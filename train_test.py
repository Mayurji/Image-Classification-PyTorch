from os import TMP_MAX
import torch
import torch.nn as nn
import numpy as np
from optimizer import optim 
from plot import trainTestPlot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class training:
    
    def __init__(self, model, optimizer, learning_rate, train_dataloader, num_epochs, test_dataloader, eval=True, plot=True, model_name=None):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.eval = eval
        self.plot = plot
        self.model_name = model_name

    def runner(self):
        
        criterion = nn.CrossEntropyLoss()
        if self.model_name in ['resnet', 'alexnet']:
            optimizer, scheduler = optim(model_name=self.model_name, model=self.model, lr=self.learning_rate)

        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
        else:
            pass
        
        train_losses = []
        train_accu = []
        test_losses = []
        test_accu = []
        # Train the model
        total_step = len(self.train_dataloader)
        for epoch in range(self.num_epochs):
            
            running_loss = 0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_loss=running_loss/len(self.train_dataloader)
                train_accuracy = 100.*correct/total
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.3f}, Train Loss: {:.4f}'
                    .format(epoch+1, self.num_epochs, i+1, total_step, train_accuracy, loss.item()))
                
                
            if self.eval:
                self.model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    running_loss = 0
                    for images, labels in self.test_dataloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = self.model(images)
                        loss= criterion(outputs,labels)
                        running_loss+=loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        test_loss=running_loss/len(self.test_dataloader)
                        test_accuracy = (correct*100)/total
                    print('Epoch: %.0f | Test Loss: %.3f | Accuracy: %.3f'%(epoch+1, test_loss, test_accuracy)) 

            if self.model_name in ['resnet', 'alexnet']:
                scheduler.step()

            train_accu.append(train_accuracy)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accu.append(test_accuracy)
    
        trainTestPlot(self.plot, train_accu, test_accu, train_losses, test_losses, self.model_name)
