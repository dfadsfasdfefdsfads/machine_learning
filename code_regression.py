from tools import *


class LogisticRegression(object):

    def Train(self, Train_data, Train_label, epoch=1, optimize='adagrad'):
        history_Train_loss = []
        self.__initParams(Train_data.shape[1])
        Train_x = Train_data.T.copy()

        for i in range(epoch):
            if (i + 1) % 100 == 0:
                print("Processing epoch: %d" % (i + 1))
            (pred, loss), (d_weight, d_beta) = self.__propagation(Train_x, Train_label)
            self.__updateParams(d_weight, d_beta, optimize)
            if i % 10 == 0:
                history_Train_loss.append(loss)
        return history_Train_loss

    def Predict(self, test_data, return_label=True):
        test_x = test_data.T.copy()
        pred = self.__sigmoid(self.weight.T @ test_x + self.beta)
        pred_label = pred.copy()
        pred_label[pred_label < 0.5] = 0
        pred_label[pred_label >= 0.5] = 1
        self.Predict_label = pred_label
        if return_label:
            return pred_label
        else:
            return pred

    def CalcAccuracy(self, test_label):
        label = test_label
        diff = self.Predict_label - label
        correct = diff[diff == 0.]
        return correct.size / label.shape[1]

    def __CrossEntropyLoss(self, pred, y):
        _, N = pred.shape
        reg_lambda = self.reg_strength
        loss = -(y * np.log(pred) + (1 - y) * np.log(1 - pred))
        loss = (1 / N) * (np.sum(loss)) + reg_lambda * np.sum(self.weight * self.weight)
        d_pred = (1 / N) * ((1 - y) * (1. / (1 - pred)) - y * (1. / pred))
        return loss, d_pred


    def __init__(self, learning_rate=5e-1, reg_strength=1e-4):
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.weight = None
        self.beta = None
        self.ada_h_w = None
        self.ada_h_b = None
        self.Predict_label = None


    def __hypothesis(self, x, backward=False, d_h=None):
        h = self.weight.T @ x + self.beta
        if backward is False:
            return h
        else:
            if d_h is None:  # calculate gradient
                d_h = np.zeros_like(h)
            d_weight = (d_h @ x.T).T
            d_weight += 2 * self.reg_strength * self.weight
            d_beta = np.sum(d_h) / x.shape[1]
            return d_weight, d_beta

    def __updateParams(self, d_weight, d_beta, optimize='adagrad'):
        if optimize is 'sgd':
            self.weight -= self.learning_rate * d_weight
            self.beta -= self.learning_rate * d_beta
        else:
            if optimize is 'adagrad':
                self.ada_h_w += d_weight * d_weight
                self.ada_h_b += d_beta * d_beta
            else:
                self.ada_h_w = 0.9 * self.ada_h_w + 0.1 * d_weight * d_weight
                self.ada_h_b = 0.9 * self.ada_h_b + 0.1 * d_beta * d_beta
            self.weight -= self.learning_rate * d_weight / (np.sqrt(self.ada_h_w) + 1e-7)
            self.beta -= self.learning_rate * d_beta / (np.sqrt(self.ada_h_b) + 1e-7)


    def __sigmoid(self, x, backward=False, dy=.0):
        y = 1 / (1 + np.exp(-x) + 1e-7)
        dx = dy * y * (1 - y)
        if backward is False:
            return y
        else:
            return dx

    def __initParams(self, dimension):
        self.weight = np.random.randn(dimension, 1)
        self.beta = 0
        self.ada_h_w = np.zeros_like(self.weight)
        self.ada_h_b = 0


    def __propagation(self, x, y):
        h = self.__hypothesis(x)
        pred = self.__sigmoid(h)
        loss, d_pred = self.__CrossEntropyLoss(pred, y)
        d_h = self.__sigmoid(h, backward=True, dy=d_pred)
        d_weight, d_beta = self.__hypothesis(x, backward=True, d_h=d_h)
        return (pred, loss), (d_weight, d_beta)


learning_rates = [5e-1, 1e-1, 5e-2, 1e-2]
reg_strengths1 = [1e-1, 1e-2]
reg_strengths2 = [1e-3, 1e-4]
epoch = 300
(Train_data, Train_label), (test_data, test_label) = getData(r"D:\mclearning\income.csv", visualize=True)

optimize = 'sgd'
best_accu_1, best_param_1 = drawLossForDiffParams(LogisticRegression, Train_data, Train_label, test_data, test_label, learning_rates, reg_strengths1, epoch, optimize=optimize)
best_accu_2, best_param_2 = drawLossForDiffParams(LogisticRegression, Train_data, Train_label, test_data, test_label, learning_rates, reg_strengths2, epoch, optimize=optimize)
best_accu = best_accu_1 if best_accu_1 > best_accu_2 else best_accu_2
best_param = best_param_1 if best_accu_1 > best_accu_2 else best_param_2
print(best_accu)
print(best_param)

epochs = [30, 50, 100, 150, 300, 500, 750, 1000, 1500]
accuracies = drawAccuOfBestForDiffEpoch(LogisticRegression, Train_data, Train_label, test_data, test_label, best_param, epochs, optimize=optimize)
print(accuracies)
