import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
from datasets import RPMSentencesRaw
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
matplotlib.use('Agg')

def visualizedata():

    save_dir = "../visualize_data/RAVEN/"
    os.makedirs(save_dir, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_gpus = torch.cuda.device_count()
    #
    # # initialize autoencoder
    # autoencoder = ResNetAutoencoder().to(device)
    #
    # if num_gpus > 1:  # use multiple GPUs
    #     autoencoder = nn.DataParallel(autoencoder)
    #
    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth')
    # autoencoder.load_state_dict(state_dict)
    # autoencoder.eval()

    # root_dir = '../pgm/neutral/'
    # train_files, _, _ = gather_files_pgm(root_dir)
    # train_files = train_files[0:32]  # delete this after test

    root_dir = '../RAVEN-10000'
    all_files = gather_files(root_dir)
    train_files = all_files[0:32]

    train_dataset = RPMSentencesRaw(train_files)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    solutions = []
    for idx, (inputs, targets) in enumerate(train_dataloader):

        solutions.extend(targets.tolist())
        images = inputs.squeeze(0)

        fig1, axs1 = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                if i==2 & j==2:
                    axs1[i,j].imshow(np.zeros([160,160]), cmap="gray")
                    axs1[i,j].axis('off')
                else:
                    axs1[i,j].imshow(images[i*3+j, :, :, :].squeeze().cpu().detach().numpy(), cmap="gray")
                    axs1[i,j].axis('off')

        fig2, axs2 = plt.subplots(2, 4)
        for i in range(2):
            for j in range(4):
                axs2[i,j].imshow(images[8 + i*4 + j, :, :, :].squeeze().cpu().detach().numpy(), cmap="gray")
                axs2[i,j].axis('off')

        save_con_path = os.path.join(save_dir, f'context_{idx}.png')
        save_can_path = os.path.join(save_dir, f'candidates_{idx}.png')
        fig1.savefig(save_con_path, bbox_inches='tight')
        fig2.savefig(save_can_path, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)

    save_sol_path = os.path.join(save_dir, 'solutions.txt')
    with open(save_sol_path, "w") as file:
        for idx, sol in enumerate(solutions):
            file.write(f"Solution to problem {idx}: {sol}\n")

def displayresults_ae():
    filepath = "../results/ae_results/v1/"
    files = os.listdir(filepath)
    random.shuffle(files)

    fig, axs = plt.subplots(5, 2)
    idx = 0
    for file in files[0:5]:
        path = os.path.join(filepath, file)
        data = np.load(path)
        image = data['image'].squeeze()
        output = data['output'].squeeze()
        axs[idx, 0].imshow(image, cmap='gray')
        axs[idx, 1].imshow(output, cmap='gray')
        idx += 1

def displayresults_tr():
    filepath = "../results/tr_results/v2"
    files = os.listdir(filepath)
    random.shuffle(files)

    guesses = []
    fig, axs = plt.subplots(5, 2)
    idx = 0
    for file in files[0:5]:
        path = os.path.join(filepath, file)
        data = np.load(path)
        image = data['guess'].squeeze()
        output = data['target'].squeeze()

        axs[idx, 0].imshow(image, cmap='gray')
        axs[idx, 1].imshow(output, cmap='gray')
        guesses.append(image)

        idx += 1

    print(np.allclose(guesses, guesses[0]*len(guesses)))

def displayresults_tr_grid():
    filepath = "../results/tr_results/v2"
    files = os.listdir(filepath)
    random.shuffle(files)

    # guesses = []
    fig1, axs1 = plt.subplots(3, 3)
    fig2, axs2 = plt.subplots(3, 3)
    fig3, axs3 = plt.subplots(1, 8)
    fig4, axs4 = plt.subplots(1,1)

    file = files[0]

    path = os.path.join(filepath, file)
    data = np.load(path)
    output_grid = data['output_image_grid']
    image_grid = data['imagetensor']
    target = data['target']

    for i in range(3):
        for j in range(3):
            axs1[i, j].imshow(output_grid[i*3 + j,:].squeeze(0), cmap='gray')
            if i==2 and j==2:
                axs2[i, j].imshow(np.zeros([160,160]), cmap='gray')
            else:
                axs2[i, j].imshow(image_grid[i*3 + j, :].squeeze(0), cmap='gray')

    for i in range(8):
        axs3[i].imshow(image_grid[8+i,:].squeeze(0), cmap='gray')

    axs4.imshow(target.squeeze(0), cmap='gray')

if __name__ == "__main__":
    visualizedata()
    # displayresults_ae()
    # displayresults_tr_grid()
    # plt.show()
    # while plt.get_fignums():
    #     plt.pause(0.1)