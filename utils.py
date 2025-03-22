import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.datasets import mnist

def lire_alpha_digit(characters=[0, 1, 2]):
    # Load dataset
    file_path = "./data/binaryalphadigs.mat"
    data = sio.loadmat(file_path)
    if 'dat' not in data:
        raise KeyError("Key 'dat' not found in the .mat file. Check dataset structure.")
    
    selected_images=data['dat'][characters[0]]
    for i in range(1,len(characters)) :
        selected_images_bis=data['dat'][characters[i]]
        selected_images=np.concatenate((selected_images,selected_images_bis),axis=0)
    n=selected_images.shape[0]
    selected_images=np.concatenate(selected_images).reshape((n,320))
    return selected_images

def lire_alpha_digit_labeled():
    """
    read all the data from the mat file

    return:
    X: the data, flattened images
    y: the labels
    """
    # load data
    file_path = "./data/binaryalphadigs.mat"
    data = sio.loadmat(file_path)
    X = data['dat']  # 36 classes x 39 samples, each is array(20, 16)
    y = np.array([i for i in range(X.shape[0]) for j in range(X.shape[1])])
    # convert to 2D array
    X = np.array([list(X[i, j].flatten())
                  for i in range(X.shape[0]) for j in range(X.shape[1])])
    # split
    n_samples = X.shape[0]
    n_train = int(0.8*n_samples)
    idx = np.random.permutation(n_samples)
    X_train = X[idx[:n_train]]
    y_train = y[idx[:n_train]]
    X_test = X[idx[n_train:]]
    y_test = y[idx[n_train:]]

    return (X_train, y_train), (X_test, y_test)

def lire_mnist_all():
    """
    read all the data from the mnist dataset

    return:
    X: the data, flattened images
    y: the labels
    """

    # load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # flatten images
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)

    # binarize
    X_train = (X_train > 127).astype(int)
    X_test = (X_test > 127).astype(int)

    return (X_train, y_train), (X_test, y_test)


def plot_generated_images(images, image_shape=(20, 16)):
    # Ensure images is a NumPy array before reshaping
    images = np.array(images)

    # Convert to 2D if images are flattened
    images = images.reshape(-1, *image_shape)

    # Determine grid size for visualization
    n_images = images.shape[0]
    print(f"Number of images: {n_images}")  # Debugging

    n_cols = int(np.ceil(np.sqrt(n_images)))  # Square-like layout
    n_rows = int(np.ceil(n_images / n_cols))

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    # Ensure axes is always iterable (handles case when n_images == 1)
    if n_images == 1:
        axes = [axes]  # Convert single Axes object into a list
    else:
        axes = axes.flatten()  # Flatten only if it's an array

    # Display each image
    for i in range(n_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')

    # Hide any unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.show()

def lire_alpha_digit_all():
    """
    read all the data from the mat file

    return:
    X: the data, flattened images
    y: the labels
    """

    # load data
    data = sio.loadmat('C:\\Users\\JLASSI\\Downloads\\Projet_dl2\\Projet_dl2\\binaryalphadigs.mat')
    X = data['dat']  # 36 classes x 39 samples, each is array(20, 16)
    y = np.array([i for i in range(X.shape[0]) for j in range(X.shape[1])])

    # convert to 2D array
    X = np.array([list(X[i, j].flatten())
                  for i in range(X.shape[0]) for j in range(X.shape[1])])

    # split
    n_samples = X.shape[0]
    n_train = int(0.8*n_samples)
    idx = np.random.permutation(n_samples)
    X_train = X[idx[:n_train]]
    y_train = y[idx[:n_train]]
    X_test = X[idx[n_train:]]
    y_test = y[idx[n_train:]]

    return (X_train, y_train), (X_test, y_test)

def deriv_sigmoid(x):
    """
    calculate the derivative of the sigmoid function

    args:
    x: sigmoid(a)

    return:
    output: the derivative sigmoid'(a) = sigmoid(a)*(1 - sigmoid(a))
    """

    return x*(1 - x)


def deriv_cross_soft(y_soft, y_true):
    """
    calculate the derivative of the cross entropy loss of the softmax

    args:
    y_true: the true labels
    y_soft: the output of the softmax (softmax(Z))

    return:
    output: the derivative of the {cross entropy + softmax} wrt Z
    """

    return y_soft-y_true
