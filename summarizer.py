from transformers import AutoTokenizer, BartForConditionalGeneration
import torch

def summarize_a_text(Text, path, source_len=1024, max_length_pred=128, min_length_pred=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = BartForConditionalGeneration.from_pretrained(path)
    model.to(device)
    model.eval()

    source = tokenizer.batch_encode_plus(
        [Text],
        max_length=source_len,
        pad_to_max_length=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        )   
    ids = source["input_ids"].to(device, dtype = torch.long)
    mask = source["attention_mask"].to(device, dtype = torch.long)

    with torch.no_grad():
        generated_ids = model.generate(
        input_ids = ids,
        attention_mask = mask, 
        max_length=max_length_pred,
        min_length=min_length_pred,
        num_beams=2,
        repetition_penalty=2.5, 
        length_penalty=1.0, 
        early_stopping=True
        )
    pred = [tokenizer.decode(g, skip_special_tokens=True, 
        clean_up_tokenization_spaces=True) for g in generated_ids]
    
    return pred[0]