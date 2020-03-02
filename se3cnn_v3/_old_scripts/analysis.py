import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


recon_loss_train = []
kld_loss_train = []
total_loss_train = []

recon_loss_val = []
kld_loss_val = []
total_loss_val = []

with open('trial_6.out', 'r') as f:
    for line in f:
        if 'recon_l' in line and 'val' not in line:
            line = line.split()
            recon_l_train = float(line[0].split('=')[1][:-1])
            kld_l_train = float(line[1].split('=')[1][:-1])
            total_l_train = float(line[2].split('=')[1])
            recon_loss_train.append(recon_l_train)
            kld_loss_train.append(kld_l_train)
            total_loss_train.append(total_l_train)
        elif 'recon_l' in line and 'val' in line:
            line = line.split()
            recon_l_val = float(line[0].split('=')[1][:-1])
            kld_l_val = float(line[1].split('=')[1][:-1])
            total_l_val = float(line[2].split('=')[1])
            recon_loss_val.append(recon_l_val)
            kld_loss_val.append(kld_l_val)
            total_loss_val.append(total_l_val)

# plt.plot(total_loss_train, label='train loss')
# plt.plot(total_loss_val, label='validation loss')
# plt.legend(loc='best')
# plt.show()

fig, axs = plt.subplots(1,2, figsize=(12,6))

axs[0].plot(recon_loss_train, label='reconstruction loss (train)')
axs[0].plot(recon_loss_val, label='reconstruction loss (validation)')
# axs[0].legend(loc='best')
axs[0].set_xlabel('Batch #')
axs[0].set_ylabel('Loss')
axs[0].set_title('Reconstruction Loss')
# plt.show()

axs[1].plot(kld_loss_train, label='train')
axs[1].plot(kld_loss_val, label='validation')
# axs[1].legend(loc='best')
axs[1].set_xlabel('Batch #')
axs[1].set_ylabel('Loss')
axs[1].set_title('KLD Loss')
plt.suptitle('Training 3DCNN_VAE')
plt.subplots_adjust(top=0.9)
plt.legend(loc='best')
plt.show()



# with open('trial_5a.out', 'r') as f:
#     for line in f:
#         if line[0] == 't':
#             line = line.split('(')
#             try:
#                 recon = float(line[1].split(',')[0])
#                 kld = float(line[2].split(',')[0])
#                 recon_loss.append(recon)
#                 kld_loss.append(kld)
#             except IndexError:
#                 pass
#         else:
#             if line[0] == '[':
#                 line = line.split()
#                 loss = float(line[1].split('=')[1])
#                 total_loss.append(loss)
#
# with open('trial_5b.out', 'r') as f:
#     for line in f:
#         if line[0] == 't':
#             line = line.split('(')
#             try:
#                 recon = float(line[1].split(',')[0])
#                 kld = float(line[2].split(',')[0])
#                 recon_loss.append(recon)
#                 kld_loss.append(kld)
#             except IndexError:
#                 pass
#         else:
#             if line[0] == '[':
#                 line = line.split()
#                 loss = float(line[1].split('=')[1])
#                 total_loss.append(loss)
#
# plt.plot(recon_loss, label='reconstruction loss')
# plt.plot(kld_loss, label='kld loss')
# plt.plot(total_loss, label='total loss')
# plt.legend(loc='best')
# plt.xlabel('Batch #')
# plt.ylabel('Loss')
# plt.title('Training Loss for 3DCNN_VAE')
# plt.show()
