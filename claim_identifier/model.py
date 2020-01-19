import torch
import torch.nn as nn 
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import pdb
from pytorch_transformers import *

class SequenceTagger(torch.nn.Module):

    def __init__(self,bert_dim,n_labels,bert_model_name,dp=0.5):
        super(SequenceTagger,self).__init__()
        bert = BertModel.from_pretrained('bert-base-german-cased')
        self.bert = bert
        self.out = torch.nn.Linear(bert_dim, n_labels)
        self.dropout = nn.Dropout(dp)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.nll_loss = nn.NLLLoss(ignore_index=-1)   
        
    def forward(self, bert_batch,true_labels,is_eval=False,is_gpu=True):
        bert_ids, bert_mask, bert_token_starts = bert_batch
        # truncate to longest sequence length in batch (usually much smaller than 512) to save GPU RAM
        max_length = (bert_mask != 0).max(0)[0].nonzero()[-1].item()+1
        if max_length < bert_ids.shape[1]: # bert_ids.shape = sentenceNumber,maxlen
            bert_ids = bert_ids[:, :max_length]
            bert_mask = bert_mask[:, :max_length]

        if is_gpu:
            device = torch.device("cuda")
            bert_ids = bert_ids.to(device)
            bert_mask = bert_mask.to(device)
            bert_token_starts = bert_token_starts.to(device)
            true_labels = [x.to(device) for x in true_labels]            
        else:
            device = torch.device("cpu")            
    
        segment_ids = torch.zeros_like(bert_mask)  # dummy segment IDs, since we only have one sentence
        attention_mask = (bert_ids != 0).long()
        attention_mask = attention_mask.to(device)
        bert_last_layer = self.bert(bert_ids, segment_ids,attention_mask=attention_mask)[0] # last_lay: sentenceNum,max_len,768

        # select the states representing each token start, for each instance in the batch
        bert_token_reprs = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(bert_last_layer, bert_token_starts)]

        # need to pad because sentence length varies
        padded_bert_token_reprs = pad_sequence(
            bert_token_reprs, batch_first=True, padding_value=-1) # padded_bert_token_reprs.shape 2x14x768
        # output/classification layer: input bert states and get log probabilities for cross entropy loss
        out = self.out(self.dropout(padded_bert_token_reprs))
        pred_logits = self.log_softmax(out) # shape pred_logits: 2x14x2 (bs x word_number x size_of_label_set)
        if is_eval:
            return pred_logits

        # Get loss for each sentence and then take average
        loss = self.nll_loss(pred_logits[0,:,:],true_labels[0])
        for i in range(1,pred_logits.shape[0]):
            tmp_lss = self.nll_loss(pred_logits[i,:,:],true_labels[i])
            loss += tmp_lss
        return loss/pred_logits.shape[0]
