## Use transformer output directly, without subsequent MLP layers
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split
from main_ae import ResNetAutoencoder, gather_files, gather_files_pgm
import time
import random
from evaluate import evaluate_model
from datasets import RPMSentencesNew, RPMSentencesRaw, CustomMNIST
from models import TransformerModelv3
import os
from torchsummary import summary
import logging

logging.basicConfig(filename='/scratch/mahirp/Projects/masked_rpm/stats.log',level=logging.INFO)
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def initialize_weights_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def main():

    # Initialize device, model
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()

    # transformer_model = TransformerModelv5(embed_dim=512, num_heads=64, abstr_depth=20, reas_depth=20, \
    #                                         cat=False).to(device)
    # transformer_model = TransformerModelMNIST(embed_dim=256, num_heads=16).to(device)
    transformer_model = TransformerModelv3(embed_dim=256, num_heads=4, con_depth=5, can_depth=8,
                                           guess_depth=20, cat=False).to(device)
    model_parameters = filter(lambda p: p.requires_grad, transformer_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params,'parameters')
    # initialize weights
    transformer_model.apply(initialize_weights_he)

    # # initialize autoencoder
    autoencoder = ResNetAutoencoder(embed_dim=256).to(device)
    autoencoder.requires_grad_(False)

    if num_gpus > 1:  # use multiple GPUs
        transformer_model = nn.DataParallel(transformer_model)
        # autoencoder = nn.DataParallel(autoencoder) # uncomment if using PGM

    # state_dict = torch.load('../modelsaves/autoencoder_v1_ep1.pth')
    state_dict = torch.load('/scratch/mahirp/Projects/masked_rpm/autoencoder_v1_ep1.pth')
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    state_dict = torch.load('/scratch/mahirp/Projects/masked_rpm/tf_v1_ep1.pth')
    transformer_model.load_state_dict(state_dict)
    ''' Load saved model '''
    # state_dict_tr = torch.load('../modelsaves/transformer_v2_ep14.pth')
    # transformer_model.load_state_dict(state_dict_tr)
    # transformer_model.eval()

    ''' Use PGM dataset '''
    # root_dir = '../pgm/neutral/'
    # train_files, val_files, test_files = gather_files_pgm(root_dir)
    # train_files = train_files[0:32] # delete this after test
    # val_files = train_files[0:32] # delete this after test

    ''' Use RAVEN dataset '''
    root_dir = '/scratch/Datasets/RPM/RAVEN-10000/'
    all_files = gather_files(root_dir)
    num_files = len(all_files)
    train_proportion = 0.7
    val_proportion = 0.15
    # test proportion is 1 - train_proportion - val_proportion
    train_files = all_files[:int(num_files * train_proportion)]
    val_files = all_files[int(num_files * train_proportion):int(num_files * (train_proportion + val_proportion))]
    # test_files = all_files[int(num_files * (train_proportion + val_proportion)):]

    # train_files = train_files[0:1500]
    # val_files = val_files[0:150]


    ''' Transformer model v2 to v4 '''
    train_dataset = RPMSentencesNew(train_files, autoencoder, device=device)
    val_dataset = RPMSentencesNew(val_files, autoencoder, device=device)

    ''' Transformer model v5 '''
    # train_dataset = RPMSentencesRaw(train_files)
    # val_dataset = RPMSentencesRaw(val_files)

    ''' MNIST transformer model '''
    # train_dataset = CustomMNIST(mnist_train, num_samples=100000)
    # val_dataset = CustomMNIST(mnist_val, num_samples=10000)

    ''' Define Hyperparameters '''
    EPOCHS = 100000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    TOTAL_DATA = len(train_dataset)  # training dataset size
    SAVES_PER_EPOCH = 10
    BATCHES_PER_SAVE = TOTAL_DATA // BATCH_SIZE // SAVES_PER_EPOCH
    VERSION = "v3-itr1"
    VERSION_SUBFOLDER = "" # e.g. "MNIST/" or ""

    ''' Instantiate data loaders, optimizer, criterion '''
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(list(transformer_model.parameters()),
                                 lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = ExponentialLR(optimizer, gamma=0.98)
    # Training loop
    train_length = len(train_dataloader)
    transformer_model.to(device)
    for epoch in range(EPOCHS):
        for idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = transformer_model(inputs) # (B,8)
            loss = criterion(outputs,targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if idx % 300 == 0:
                autoencoder.eval()
                with torch.no_grad():
                    val_i = 0
                    break_point = random.randint(150, 200)
                    num_correct = 0
                    total = 0
                    for _, (inputs,targets) in enumerate(val_dataloader):
                        val_i+=1
                        total+=list(inputs.size())[0]
                        inputs = inputs.to(device)
                        targets = torch.argmax(targets.to(device),dim=1)
                        outputs = transformer_model(inputs)  # (batch_size,8)
                        guesses = torch.argmax(outputs, dim=1)
                        num_correct += torch.eq(guesses, targets).sum().item()
                        if val_i==break_point:
                            break
                    val_loss = (num_correct / (total)) * 100
            if idx % 50 == 0:
                print(f"\repoch {epoch} - {idx}/{train_length}: loss : {loss.item()} lr :{scheduler.get_last_lr()[0]} val: {val_loss}",
                      end='')
            if idx % 150 == 0:
                logging.info(f"epoch {epoch} - {idx}/{train_length}: loss : {loss.item()} lr :{scheduler.get_last_lr()[0]} val: {val_loss}")
        print('\n')
        if epoch%SAVES_PER_EPOCH==0:
            scheduler.step()
            torch.save(transformer_model.state_dict(), f"/scratch/mahirp/Projects/masked_rpm/tf_v1_ep{epoch + 1}.pth")
    #     print(f"Epoch {epoch+1}/{EPOCHS} completed: loss = {loss.item()}\n")
    #
    # # Evaluate the model
    # proportion_correct = evaluate_model(transformer_model, val_dataloader, device=device)
    # print(f"Proportion of answers correct: {proportion_correct}")
    #
    # output_file_path = f"../tr_results/{VERSION}/{VERSION_SUBFOLDER}proportion_correct_test.txt"
    # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # with open(output_file_path, "w") as file:
    #     file.write(f"Proportion of answers correct: {proportion_correct}.")

if __name__ == "__main__":
    main()