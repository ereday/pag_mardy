from pytorch_transformers import *
import torch
import re
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
import pdb

class BertForMultiLabelSequenceClassification(torch.nn.Module):
    def __init__(self, args):
        super(BertForMultiLabelSequenceClassification, self).__init__()
        try:
            self.num_labels = args['num_labels']
            bert = BertModel.from_pretrained('bert-base-german-cased')            
            self.bert = bert
            self.dropout = torch.nn.Dropout(args['dp'])            
            self.classifier = torch.nn.Linear(args['hs'], args['num_labels'])
            #self.apply(self.init_bert_weights)        
        except:
            self.num_labels = args.num_labels
            bert = BertModel.from_pretrained('bert-base-german-cased')
            self.bert = bert
            self.dropout = torch.nn.Dropout(args.dp)            
            self.classifier = torch.nn.Linear(args.hs, args.num_labels)
            

    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _,pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)        
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            #pdb.set_trace()
            return loss
        else:
            return logits
        





