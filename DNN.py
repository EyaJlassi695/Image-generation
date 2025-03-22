import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import DBN, RBM  # Assuming RBM & DBN are correctly defined in models.py
from utils import lire_alpha_digit_all, deriv_sigmoid, deriv_cross_soft
import numpy as np
import matplotlib.pyplot as plt

class DNN(DBN):
    def __init__(self, layer_sizes):
        super().__init__(layer_sizes[:-1])  # Use DBN's initialization

        self.dim_output = layer_sizes[-1]  # Number of output classes
        self.b_out = np.zeros((1, self.dim_output))
        self.W_out = np.random.randn(self.layer_sizes[-1], self.dim_output)*0.1

    def calcul_softmax(self, input, W, b):  # ✅ FIX: Added `self`
        output = np.dot(input, W) + b
        output = np.exp(output - np.max(output, axis=1, keepdims=True))  # ✅ Numerical stability fix
        output = output / np.sum(output, axis=1, keepdims=True)
        return output

        return output
    def pretrain_DNN(self, training_data,learning_rate, batch_size, n_epochs_per_layer,verbose=False):
        self.train_DBN(training_data, learning_rate, batch_size, n_epochs_per_layer, verbose)

    def entree_sortie_reseau(self, input):
        all_layers_list = [input]
        for i in range(self.L):
            h = self.layers[i].entree_sortie_RBM(all_layers_list[i])
            all_layers_list.append(h)

        output = self.calcul_softmax(h, self.W_out, self.b_out)
        all_layers_list.append(output)

        return all_layers_list

    def retropropagation(self, data, target, batch_size, n_epochs, lr):
        # one-hot encode the target
        target = np.eye(self.dim_output)[target]

        for epoch in range(n_epochs):
            for i_start in range(0, data.shape[0], batch_size):
                i_end = min(data.shape[0], i_start + batch_size)
                batch = data[i_start:i_end]
                batch_target = target[i_start:i_end]

                # forward pass
                all_layers_list = self.entree_sortie_reseau(batch)
                output = all_layers_list[-1]

                # loss
                cross_entropy = -np.sum(batch_target*np.log(output), axis=1)
                loss = np.mean(cross_entropy)

                # backpropagation (delta is used to compute gradients)
                delta = [deriv_cross_soft(output, y_true=batch_target)]
                delta = [np.dot(delta[0], self.W_out.T) *
                         deriv_sigmoid(all_layers_list[-2])] + delta
                for i in range(3, self.L+2):
                    delta = [np.dot(delta[0], self.layers[-i+2].W.T) *
                             deriv_sigmoid(all_layers_list[-i])] + delta
                # len(delta) is self.L+1, first is input transform

                # update
                grad_W_out = np.dot(all_layers_list[-2].T, delta[-1])
                grad_W_out = grad_W_out / (i_end - i_start)
                grad_b_out = np.mean(delta[-1], axis=0)

                self.W_out -= lr*grad_W_out
                self.b_out -= lr*grad_b_out

                for i in range(self.L):
                    grad_W = np.dot(all_layers_list[i].T, delta[i])
                    grad_W = grad_W / (i_end - i_start)
                    grad_b = np.mean(delta[i], axis=0)

                    self.layers[i].W -= lr*grad_W
                    self.layers[i].b -= lr*grad_b

            # show epoch error
            print(f'Epoch {epoch}, Cross-Entropy Loss: {loss}')
            # Show epoch error



    def test_DNN(self, data, target):
        output = self.entree_sortie_reseau(data)[-1]
        pred = np.argmax(output, axis=1)

        accuracy = np.mean(pred == target)

        return accuracy


# dbn structure
dim_hidden = 100
n_layers = 5
dims = n_layers * [dim_hidden]

# training parameters
pretrain = True
batch_size = 32
lr = 0.1
n_epochs_pretrain = 300
n_epochs_train = 500

# load alpha digit data
(X_train, y_train), (X_test, y_test) = lire_alpha_digit_all()

n_classes = len(np.unique(y_train))

# train
dnn = DNN([X_train.shape[1], *dims, n_classes])
if pretrain:
    print('Pretraining...')
    dnn.pretrain_DNN(X_train,lr, batch_size, n_epochs_pretrain)
print('Training...')
dnn.retropropagation(X_train, y_train,batch_size,n_epochs_train,lr )


# test
acc = dnn.test_DNN(X_test, y_test)
print(f'Test accuracy: {acc:.2f}')
