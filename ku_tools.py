from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def getData(csv_file_path, keep_dim=True, visualize=False):
    if os.path.exists(csv_file_path) is False:
        print(csv_file_path + "Not found.")
        return None

    features_all = []
    labels_all = []

    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            features_all.append([float(feature) for feature in row[1:58]])
            labels_all.append(int(row[-1]))

    data = np.array(features_all)
    label = np.array(labels_all)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / data_std

    Train_data = data[:3000]
    test_data = data[3000:]
    Train_label = label[:3000]
    test_label = label[3000:]
    if keep_dim:
        Train_label = Train_label.reshape((1, -1))
        test_label = test_label.reshape((1, -1))
    if visualize:
        t_sne(data, label, 'visualization of dataset')

    return (Train_data, Train_label), (test_data, test_label)


def drawLossForDiffParams(model, Train_data, Train_label, test_data, test_label,
                          learning_rates, reg_strengths, epoch, optimize='sgd'):
    best_accu = .0
    best_param = {'learning_rate': .0, 'regularization': .0}
    index = [i * 10 for i in range(int(epoch / 10))]

    for reg_strength in reg_strengths:
        for learning_rate in learning_rates:
            lr = model(learning_rate=learning_rate, reg_strength=reg_strength)
            history_Train_loss = lr.Train(Train_data, Train_label, epoch=epoch, optimize=optimize)
            lr.Predict(test_data)
            accu = lr.CalcAccuracy(test_label)
            # draw loss
            plt.plot(index, history_Train_loss,
                     marker='o', label='train_loss ' + 'lr: ' + str(learning_rate) + ' reg: ' + str(reg_strength))
            # select best param combination
            if accu > best_accu:
                best_accu = accu
                best_param['learning_rate'] = learning_rate
                best_param['regularization'] = reg_strength
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss of different hyper-params')
    plt.grid()
    plt.show()
    return best_accu, best_param


def drawAccuOfBestForDiffEpoch(model, Train_data, Train_label, test_data, test_label,
                               best_param, epochs, optimize='sgd'):
    learning_rate = best_param['learning_rate']
    reg_strength = best_param['regularization']
    accuracies = []
    lr = model(learning_rate=learning_rate, reg_strength=reg_strength)
    for epoch in epochs:
        lr.Train(Train_data, Train_label, epoch=epoch, optimize=optimize)
        lr.Predict(test_data)
        accu = lr.CalcAccuracy(test_label)
        accuracies.append(accu)
    plt.plot(epochs, accuracies, 'ro-')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy of best classifier by different epoch')
    plt.grid()
    plt.show()
    return accuracies


def t_sne(x, labels, fig_title=None):
    tsne = manifold.TSNE(n_components=3, init='pca')
    x_tsne = tsne.fit_transform(x)
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue']
    for i in range(int(x.shape[0]*0.4)):
        ax.scatter(x_tsne[i, 0], x_tsne[i, 1],x_tsne[i, 2],
                   color=colors[labels[i]])
    if fig_title is not None:
        plt.title(fig_title)
    plt.show()

