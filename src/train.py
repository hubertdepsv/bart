# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')
import torch
from tqdm.auto import tqdm


def train(model, training_loader, tokenizer, optimizer, device, lr_scheduler=None):

    """
    Function to be called for training with the parameters passed from main function

    """
    model.train()  # Set model to training mode
    loss_train = []
    for data in tqdm(training_loader):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels
        )
        
        loss = outputs[0]
        loss_train.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
    
    return loss_train