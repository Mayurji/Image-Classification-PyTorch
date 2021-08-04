from pathlib import Path
import matplotlib.pyplot as plt

def trainTestPlot(plot, train_accu, test_accu, train_losses, test_losses, model_name):

    if plot:
        Path('plot/').mkdir(parents=True, exist_ok=True)
        plot1 = plt.figure(1)
        plt.plot(train_accu, '-o')
        plt.plot(test_accu, '-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Test'])
        plt.title('Train vs Test Accuracy')            
        plt.savefig('plot/'+model_name+'_train_test_acc.png')

        plot2 = plt.figure(2)
        plt.plot(train_losses,'-o')
        plt.plot(test_losses,'-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Test'])
        plt.title('Train vs Test Losses')
        plt.savefig('plot/'+model_name+'_train_test_loss.png')