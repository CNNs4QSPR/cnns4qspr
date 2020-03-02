import numpy as np
import matplotlib.pyplot as plt

### Load Data
datapath = '../networks/SE3ResNet34Small/data/'

total_loss_tr = np.load(datapath+'loss_avg.npy')
BCE_loss_tr = np.load(datapath+'loss_avg_BCE.npy')
KLD_loss_tr = np.load(datapath+'loss_avg_KLD.npy')

total_loss_val = np.load(datapath+'loss_avg_val.npy')
BCE_loss_val = np.load(datapath+'loss_avg_BCE_val.npy')
KLD_loss_val = np.load(datapath+'loss_avg_KLD_val.npy')

latent_means_tr = np.load(datapath+'latent_mean_outs.npy')
latent_vars_tr = np.load(datapath+'latent_var_outs.npy')
training_labels_tr = np.load(datapath+'training_labels.npy')

latent_means_val = np.load(datapath+'latent_mean_outs_val.npy')
latent_vars_val = np.load(datapath+'latent_var_outs_val.npy')
training_labels_val = np.load(datapath+'training_labels_val.npy')

### Loss plot from numpy arrays
# fig, axs = plt.subplots(1,3,figsize=(12,6))
#
# axs[0].plot(BCE_loss_tr, alpha=0.75, ls=':', label='Train')
# axs[0].plot(BCE_loss_val, alpha=0.75, ls=':', label='Validation')
# axs[0].set_xlabel('Epoch')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('VAE Reconstruction Loss')
#
# axs[1].plot(KLD_loss_tr, alpha=0.75, ls=':', label='Train')
# axs[1].plot(KLD_loss_val, alpha=0.75, ls=':', label='Validation')
# axs[1].set_xlabel('Epoch')
# axs[1].set_title('KL Divergence Loss')
#
# axs[2].plot(total_loss_tr, alpha=0.75, ls=':', label='Train')
# axs[2].plot(total_loss_val, alpha=0.75, ls=':', label='Validation')
# axs[2].set_xlabel('Epoch')
# axs[2].set_title('Total Loss')
#
# plt.legend(loc='best')
# plt.show()

print(latent_means_val.shape)
print(training_labels_val.shape)



### Loss plot from slurm file
# recon_loss_train = []
# kld_loss_train = []
# total_loss_train = []
#
# recon_loss_val = []
# kld_loss_val = []
# total_loss_val = []
#
# with open('../_old_runs/trial_7.out', 'r') as f:
#     for line in f:
#         if 'recon_l' in line and 'val' not in line:
#             line = line.split()
#             recon_l_train = float(line[0].split('=')[1][:-1])
#             kld_l_train = float(line[1].split('=')[1][:-1])
#             total_l_train = float(line[2].split('=')[1])
#             recon_loss_train.append(recon_l_train)
#             kld_loss_train.append(kld_l_train)
#             total_loss_train.append(total_l_train)
#         elif 'recon_l' in line and 'val' in line:
#             line = line.split()
#             recon_l_val = float(line[0].split('=')[1][:-1])
#             kld_l_val = float(line[1].split('=')[1][:-1])
#             total_l_val = float(line[2].split('=')[1])
#             recon_loss_val.append(recon_l_val)
#             kld_loss_val.append(kld_l_val)
#             total_loss_val.append(total_l_val)
#
# fig, axs = plt.subplots(1,2, figsize=(12,6))
#
# axs[0].plot(recon_loss_train, label='reconstruction loss (train)')
# axs[0].plot(recon_loss_val, label='reconstruction loss (validation)')
# # axs[0].legend(loc='best')
# axs[0].set_xlabel('Batch #')
# axs[0].set_ylabel('Loss')
# axs[0].set_title('Reconstruction Loss')
# # plt.show()
#
# axs[1].plot(kld_loss_train, label='train')
# axs[1].plot(kld_loss_val, label='validation')
# # axs[1].legend(loc='best')
# axs[1].set_xlabel('Batch #')
# axs[1].set_ylabel('Loss')
# axs[1].set_title('KLD Loss')
# plt.suptitle('Training 3DCNN_VAE')
# plt.subplots_adjust(top=0.9)
# plt.legend(loc='best')
# plt.show()
