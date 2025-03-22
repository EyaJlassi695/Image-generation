import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from utils import lire_alpha_digit,plot_generated_images

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
            v=v.reshape((20,16))
            images.append(v)
        return images



# Extract training data
X_train = lire_alpha_digit([10,11,12])
dim_visible = X_train.shape[1]  # Infer p from X
dim_hidden = 500  # Hidden layer size 

# Initialize RBM
rbm = RBM(dim_visible, dim_hidden)

# Train the model
rbm.train_RBM(X_train, learning_rate=10**(-2), batch_size=10, n_epochs=1000, verbose=False)

generated_images = rbm.generer_image_RBM(nb_images=5, gibbs_steps=200)  # Take only the first generated image

plot_generated_images(generated_images,image_shape=(20, 16))

