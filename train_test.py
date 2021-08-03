import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class training:
    
    def __init__(self, model, optimizer, learning_rate, train_dataloader, num_epochs, test_dataloader, eval=True):
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs
        self.eval = eval

    def train(self):
        
        criterion = nn.CrossEntropyLoss()
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            pass

        # Train the model
        total_step = len(self.train_dataloader)
        for epoch in range(self.num_epochs):
            
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

                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
                    # Loss.append(loss.cpu().detach().numpy())
                    # visual.plot_loss(np.mean(Loss), i)
                    # Loss.clear()
            

        if self.eval:
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in self.test_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print(f'Accuracy: {(correct*100)/total}')

