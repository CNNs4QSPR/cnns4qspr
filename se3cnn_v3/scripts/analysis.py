import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import imageio

datapath = '../networks/VAE/data/'
train_accs = np.load(datapath+'accs.npy')
val_accs = np.load(datapath+'val_accs.npy')
train_labels = np.load(datapath+'train_labels.npy')
val_labels = np.load(datapath+'val_labels.npy')
train_latent_space = np.load(datapath+'train_latent_space.npy')
val_latent_space = np.load(datapath+'val_latent_space.npy')

### Training progress
# plt.plot(train_accs, label='Training Accuracy')
# plt.plot(val_accs, label='Validation Accuracy')
# plt.legend(loc='best')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()

color_palette = ['#377856', '#3F5D8F', '#DAA190']

images = []
os.mkdir('gif')
for i in range(train_latent_space.shape[0]):
    latent_space = train_latent_space[i,:,:]
    labels = train_labels[i,:]
    pca = PCA(2)
    ics = pca.fit_transform(latent_space)

    ### 2D
    plt.scatter(ics[np.where(labels == 0.),0], ics[np.where(labels == 0.),1], c=color_palette[0], s=2.5, alpha=0.3, label='Domain 1')
    plt.scatter(ics[np.where(labels == 1.),0], ics[np.where(labels == 1.),1], c=color_palette[1], s=2.5, alpha=0.3, label='Domain 2')
    plt.scatter(ics[np.where(labels == 2.),0], ics[np.where(labels == 2.),1], c=color_palette[2], s=2.5, alpha=0.3, label='Domain 3')
    plt.xlim([-.25,.25])
    plt.ylim([-.25,.25])
    plt.xlabel('IC 1')
    plt.ylabel('IC 2')
    plt.title('Epoch {}'.format(i+1))
    plt.savefig('gif/{}.png'.format(i))
    plt.close()
    images.append(imageio.imread('gif/{}.png'.format(i)))
imageio.mimsave('latent2_reorg.gif', images)

### 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(ics[np.where(labels == 0.),0], ics[np.where(labels == 0.),1], ics[np.where(labels == 0.),2], c=color_palette[0], s=2.5, alpha=0.5)
# ax.scatter(ics[np.where(labels == 1.),0], ics[np.where(labels == 1.),1], ics[np.where(labels == 1.),2], c=color_palette[1], s=2.5, alpha=0.5)
# ax.scatter(ics[np.where(labels == 2.),0], ics[np.where(labels == 2.),1], ics[np.where(labels == 2.),2], c=color_palette[2], s=2.5, alpha=0.5)
# plt.show()
