import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from utils import lire_alpha_digit,plot_generated_images
from models import RBM


class DBN:
    def __init__(self, layer_sizes):
        self.layers = [RBM(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def train_DBN(self, training_set, learning_rate, batch_size, n_epochs_per_layer, verbose=False):
        input_data = training_set
        for i, rbm in enumerate(self.layers):
            print(f"Training RBM {i+1}/{len(self.layers)}")
            rbm.train_RBM(input_data, learning_rate, batch_size, n_epochs_per_layer, verbose)
            input_data = rbm.entree_sortie_RBM(input_data)

    def generer_image_DBN(self, nb_images, gibbs_steps):
        p = self.layers[0].W.shape[0]  # Input layer size
        images = []
        size_img=(20,16)
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
    
# Define DBN structure
X_train = lire_alpha_digit([10])
dim_visible = X_train.shape[1]  # Infer p from X
layer_sizes = [dim_visible, 500, 250, 100]  # Example structure

# Initialize and train DBN
dbn = DBN(layer_sizes)
dbn.train_DBN(X_train, learning_rate=0.01, batch_size=10, n_epochs_per_layer=100, verbose=True)

# Generate and visualize images
images= dbn.generer_image_DBN(nb_images=5, gibbs_steps=300)
plot_generated_images(images)