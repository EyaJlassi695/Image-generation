import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from utils import *
class RBM:
    def __init__(self,dim_visible, dim_hidden):
        self.a = np.zeros((1, dim_visible))
        self.b = np.zeros((1, dim_hidden))
        self.W = np.random.randn(dim_visible, dim_hidden)*0.1

    def entree_sortie_RBM(self, V):
        V = torch.tensor(V, dtype=torch.float32)  # Convert to tensor
        W = torch.tensor(self.W, dtype=torch.float32)
        b = torch.tensor(self.b, dtype=torch.float32)
        return torch.sigmoid(V @ W + b).numpy()  # Convert back to NumPy for further operations

    def sortie_entree_RBM(self, H):
        H = torch.tensor(H, dtype=torch.float32)  # Convert to tensor
        W = torch.tensor(self.W, dtype=torch.float32)
        a = torch.tensor(self.a, dtype=torch.float32)
        return torch.sigmoid(H @ W.T + a).numpy()  # Convert back to NumPy for further operations

    def train_RBM(self, training_set, learning_rate, batch_size, n_epochs, verbose=False):
        p, q = self.W.shape

        weights = []
        losses = []

        for i in range(n_epochs):
            n = training_set.shape[0]
            for i_batch in range(0, n, batch_size):
                train_batch = training_set[i_batch:min(i_batch + batch_size, n), :]
                t_batch_i = train_batch.shape[0]  # Can be < to len_batch at the last iteration

                V0 = copy.deepcopy(train_batch)
                pH_V0 = self.entree_sortie_RBM(V0)
                H0 = np.random.binomial(1, pH_V0)   # Hidden features
                pV_H0 = self.sortie_entree_RBM(H0)
                V1 = np.random.binomial(1, pV_H0)   # Reconstruction of V0
                pH_V1 = self.entree_sortie_RBM(V1)

                da = np.mean(V0 - V1, axis=0)
                db = np.mean(pH_V0 - pH_V1, axis=0)
                dW = V0.T @ pH_V0 - V1.T @ pH_V1
                dW = dW/(t_batch_i)

                self.a += learning_rate * da
                self.b += learning_rate * db
                self.W += learning_rate * dW


            # Reconstruction error
            H = self.entree_sortie_RBM(training_set)
            training_set_rec = self.sortie_entree_RBM(H)
            loss = np.mean((training_set - training_set_rec) ** 2)
            if i % 10 == 0 and verbose:
                print("epoch " + str(i) + "/" + str(n_epochs) + " - loss : " + str(loss))


    def generer_image_RBM(self, nb_images, gibbs_steps):
        p,q=self.W.shape
        
        images = []
        for i in range(nb_images):#Pour chaque image faire un gipps
            v=(np.random.rand(p)<0.5)*1
            for j in range(gibbs_steps):
                h = (np.random.rand(q)<self.entree_sortie_RBM(v))*1
                v = (np.random.rand(p)<self.sortie_entree_RBM(h))*1
            v=v.reshape((28,28))
            images.append(v)
        return images

class DBN:
    def __init__(self, layer_sizes):
        self.L = len(layer_sizes) - 1  # number of hidden layers
        self.layer_sizes = layer_sizes
        self.layers = [RBM(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def train_DBN(self, training_set, learning_rate, batch_size, n_epochs_per_layer, verbose=False):
        for i in range(self.L):
            print(f'Training layer {i+1}')
            self.layers[i].train_RBM(training_set, learning_rate, batch_size, n_epochs_per_layer, verbose)
            training_set = self.layers[i].entree_sortie_RBM(training_set)

    def generer_image_DBN(self, nb_images, gibbs_steps, size_img=(28, 28)):
        p = self.layers[0].W.shape[0]  # Input layer size
        images = []
        for _ in range(nb_images):
            # Initialize a random visible vector
            v = (np.random.rand(1, p) < 0.5).astype(np.float32)  # Ensure (1, p) shape
            
            for _ in range(gibbs_steps):
                # Propagate upwards through the DBN (visible → hidden)
                for rbm in self.layers:
                    v = rbm.entree_sortie_RBM(v)  # Hidden activation
                    v = (np.random.rand(*v.shape) < v).astype(np.float32)  # Sample binary hidden state

                # Propagate downwards through the DBN (hidden → visible)
                for rbm in reversed(self.layers):
                    v = rbm.sortie_entree_RBM(v)  # Visible activation
                    v = (np.random.rand(*v.shape) < v).astype(np.float32)  # Sample binary visible state
            
            # Reshape the final visible layer back to an image
            images.append(v.reshape(size_img))
        return images

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