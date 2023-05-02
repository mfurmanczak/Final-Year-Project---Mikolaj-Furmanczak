import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


# Change this to the model you want to analyze
model_num = 1


def show_results(kfold, history_pd):
    average = (np.mean(history_pd['val_accuracy'][-5:]) * 100)
    max_accuracy = (np.max(history_pd['val_accuracy']) * 100)
    min_loss = (np.min(history_pd['val_loss']))

    average_training = (np.mean(history_pd['accuracy'][-5:]) * 100)
    max_accuracy_training = (np.max(history_pd['accuracy']) * 100)
    min_loss_training = (np.min(history_pd['loss']))

    print('\tMax training accuracy: %.4f %%' % max_accuracy_training)
    print('\tMin training loss: %.5f' % min_loss_training)
    print('\tAverage accuracy on the training set: %.4f %%' % average_training)

    print('\tMax validation accuracy: %.4f %%' % max_accuracy)
    print('\tMin validation loss: %.5f' % min_loss)
    print('\tAverage accuracy on the validation set: %.4f %%' % average)
    print(kfold)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_pd['accuracy'], label='Training accuracy')
    plt.plot(history_pd['val_accuracy'], label='Validation accuracy')
    plt.title('Training and validation accuracy for: Fold' + str(kfold))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history_pd['loss'], label='Training loss')
    plt.plot(history_pd['val_loss'], label='Validation loss')
    plt.title('Training and validation loss for: Fold' + str(kfold))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()

    return average, kfold


def main():
    histories = glob('history/Model ' + str(model_num) + '/*.csv')
    averages = []
    all_averages = []
    for i, history in enumerate(histories):
        history_df = pd.read_csv(history)
        average_max_fold, kfold = show_results(i+1, history_df)

        averages.append(average_max_fold)

    total_average = np.mean(averages)
    max_average = np.max(averages)
    print(averages)
    print('Max average accuracy: %.2f %%' %
          max_average + ' for fold: ' + str(np.argmax(averages)+1))
    print('Total average accuracy: %.2f %%' % total_average)


if __name__ == "__main__":
    main()
