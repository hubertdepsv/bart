# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger('main_logger')
import torch
from tqdm.auto import tqdm


def validate(model, val_loader, tokenizer, metric, device, summ_len):
    """
    Function to be called for validating with the parameters passed from main function

    """

    model.eval()
    precision_val = []
    recall_val = []
    fmeasure_val = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=summ_len,
                min_length=30,
                num_beams=2, # Number of beams for beam search
                repetition_penalty=2.5, #repetition penality
                early_stopping=True
            )
            
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

            #ROUGE 
            rouge_dict = metric.compute(predictions=preds, references=target)
            # Mid
            precision_val.append(rouge_dict['rouge2'][1][0])
            recall_val.append(rouge_dict['rouge2'][1][1])
            fmeasure_val.append(rouge_dict['rouge2'][1][2])
    
    return precision_val, recall_val, fmeasure_val
