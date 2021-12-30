# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')
from train import train
from validate import validate
from dataframe import dataframe_to_pandas, DataSetClass

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration 
from datasets import load_metric, load_dataset


def summarizer_rainer(conf):

    """
    Summarizer trainer

    """

    #Setting up the device for GPU usage
    device = 'cuda' if cuda.is_available() else 'cpu'

    #Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(conf['model_checkpoint'])
    model = BartForConditionalGeneration.from_pretrained(conf['model_checkpoint'])
    model = model.to(device)    

    #Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=conf['lr']
    )

    #Definig lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500,
                            T_mult=1,
                            eta_min=1e-6,
                            last_epoch=-1,
                            verbose=False)

    #Defining the ROUGE metric
    metric = load_metric("rouge")

    raw_datasets = load_dataset(conf["dataset_name"])
    train_df = dataframe_to_pandas(raw_datasets["train"], conf['train_len'])
    validation_df = dataframe_to_pandas(raw_datasets["validation"], conf['validate_len'])

    # DATALOADER
    # Creation of Dataloaders for training. 
    training_set = DataSetClass(
        train_df,
        tokenizer,
        conf['source_len'],
        conf['summ_len'],
        conf['source_text'],
        conf['target_text']
        )
    training_loader = DataLoader(training_set, batch_size=conf['train_batch_size'], 
        shuffle=True, num_workers=0)

    # Creation of Dataloaders for valdifation.
    val_set = DataSetClass(
        validation_df,
        tokenizer,
        conf['source_len'],
        conf['summ_len'],
        conf['source_text'],
        conf['target_text']
        )
    val_loader = DataLoader(val_set, batch_size=conf['validate_batch_size'],
        shuffle=False, num_workers=0)


    history_loss_train = []
    history_precision_val = []
    history_recall_val = []
    history_fmeasure_val = []
    list_train_lr = []

    for epoch in range(conf['num_epochs']):
        
        logging.print('-' * 25)
        logging.print('Epoch {}/{}'.format(epoch + 1, conf['num_epochs']))
        logging.print('-' * 25)
    
        #TRAINING PART 
        logging.print('TRAINING PART')
        logging.print('-' * 10)

        if conf['scheduler'] == True :
            loss_train = train(model, training_loader, tokenizer, optimizer,
             device, lr_scheduler=lr_scheduler)
        else :
            loss_train = train(model, training_loader, tokenizer, optimizer, device)
            
        history_loss_train.append(np.mean(loss_train))
        logging.print('Mean train loss for epoch {}: {}'.format(epoch + 1, np.mean(loss_train)))
    
        # Eval Part
        print('-' * 10)
        logging.print('EVALUATION PART')
        logging.print('-' * 10)
        
        precision_val, recall_val, fmeasure_val = validate(model, val_loader, tokenizer, 
            metric, device, conf['summ_len'])

        history_precision_val.append(precision_val)
        history_recall_val.append(recall_val)
        history_fmeasure_val.append(fmeasure_val)
        logging.print('Mean rouge precision for epoch {}: {}'.format(epoch + 1, np.mean(precision_val)))
        logging.print('Mean rouge recall for epoch {}: {}'.format(epoch + 1, np.mean(recall_val)))
        logging.print('Mean rouge fmeasure for epoch {}: {}'.format(epoch + 1, np.mean(fmeasure_val)))

    
    #Save the model and tokenizer 
    model.save_pretrained(conf['path'])
    tokenizer.save_pretrained(conf['path'])

    #Create and save the training graph
    fig, ax = plt.subplots(1,2, figsize=(24,12))

    k = len(history_loss_train)

    ax[0].plot(range(k), history_loss_train, label='train loss')
    ax[0].set(xlabel='number of batchs', ylabel='accuracy',
        title='Evolution of loss threw batch (Adam optimizer)')
    ax[0].legend()

    k = len(history_precision_val)
    ax[1].plot(range(k), history_precision_val, label='val precision')
    ax[1].plot(range(k), history_recall_val, label='val recall')
    ax[1].plot(range(k), history_fmeasure_val, label='val fmeasure')
    ax[1].set(xlabel='number of epoch', ylabel='loss',
        title='Evolution of ROUGE threw batch')
    ax[1].legend()
    plt.savefig(os.path.join(conf['path'], "training_fig.png"))


            