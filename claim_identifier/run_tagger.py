import numpy as np
from model import *
from utils import *
import argparse
import codecs
import json
import pdb
MASK = "[MASK]"
CLS = "[CLS]"
SEP = "[SEP]"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.manual_seed(1)
torch.cuda.manual_seed(1)


def subword_tokenize(tokenizer,tokens):
    """Segment each token into subwords while keeping track of
        token boundaries.
        Parameters
        ----------
        tokens: A sequence of strings, representing input tokens.
        Returns
        -------
        A tuple consisting of:
            - A list of subwords, flanked by the special symbols required
                by Bert (CLS and SEP).
            - An array of indices into the list of subwords, indicating
                that the corresponding subword is the start of a new
                token. For example, [1, 3, 4, 7] means that the subwords
                1, 3, 4, 7 are token starts, while all other subwords
                (0, 2, 5, 6, 8...) are in or at the end of tokens.
                This list allows selecting Bert hidden states that
                represent tokens, which is necessary in sequence
                labeling.
    """
    subwords = list(map(tokenizer.tokenize, tokens))
    subword_lengths = list(map(len, subwords))
    subwords = [CLS] + list(flatten(subwords)) + [SEP]
    token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])
    return subwords, token_start_idxs


def convert_tokens_to_ids(tokenizer,tokens, max_len,pad=True):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    #ids = torch.tensor([token_ids]).to(device=device)
    ids = torch.tensor([token_ids])
    assert ids.size(1) < max_len
    if pad:
        padded_ids = torch.zeros(1, max_len).to(ids)
        padded_ids[0, :ids.size(1)] = ids
        mask = torch.zeros(1, max_len).to(ids)
        mask[0, :ids.size(1)] = 1
        return padded_ids, mask
    else:
        return ids
        
def subword_tokenize_to_ids(tokenizer,tokens,max_len):
    subwords, token_start_idxs = subword_tokenize(tokenizer,tokens)
    subword_ids, mask = convert_tokens_to_ids(tokenizer,subwords,max_len)
    token_starts = torch.zeros(1, max_len).to(subword_ids)
    token_starts[0, token_start_idxs] = 1
    return subword_ids, mask, token_starts


    
def flatten(list_of_lists):
    for list in list_of_lists:
        for item in list:
            yield item


            
def collate_fn(featurized_sentences):
    bert_batch = [torch.cat([features[key] for features in featurized_sentences], dim=0) for key in
                  ("bert_ids", "bert_mask", "bert_token_starts")]
    return bert_batch



#max_len = 30
#sentences=  [['Bis', '2013', 'steigen', 'die', 'Mittel', 'aus', 'dem', 'EU-Budget',
#              'auf', 'rund', '120', 'Millionen', 'Euro', '.'],['Slowenien', 'will', 'Zaun', 'zu', 'Kroatien']]
#sentences=  [['Slowenien', 'will', 'Zaun', 'zu', 'Kroatien']]
def get_data(sentences,tokenizer,max_len):    
    featurized_sentences = []
    for tokens in sentences:
        feats = {}
        feats["bert_ids"],feats["bert_mask"],feats["bert_token_starts"] = subword_tokenize_to_ids(tokenizer,tokens,max_len)
        featurized_sentences.append(feats)
    return featurized_sentences


def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

def _read_data(fn, verbose=True, column_no=-1):
    word_sequences = list()
    tag_sequences = list()
    curr_words = list()
    curr_tags = list()
    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    for k, line in enumerate(lines):
        elements = line.strip().split('\t')
        if len(elements) < 3: # end of the document
            word_sequences.append(curr_words)
            tag_sequences.append(curr_tags)
            curr_words = list()
            curr_tags = list()
            continue
        word = elements[1]
        tag = elements[2].split(':')[0]
        curr_words.append(word)
        curr_tags.append(tag)
    if verbose:
        print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
    return word_sequences, tag_sequences    

def read_data(file_dir,split_name,l2ix):
    sentences,labels = _read_data(file_dir+'/'+split_name)    
    return sentences,[[l2ix[token_label] for token_label in sentence_labels] for sentence_labels in labels ]

def read_gen_data(file_dir,l2ix):
    sentences,labels = _read_data(file_dir)    
    return sentences,[[l2ix[token_label] for token_label in sentence_labels] for sentence_labels in labels ]

def batch_labels(labels,bs,ignore_index=-1):
    batches = []
    for i in range(0,len(labels),bs):
        batch = labels[i:min(len(labels),i+bs)]
        max_len = max([len(x) for x in batch])
        for j in range(len(batch)):
            diff = max_len - len(batch[j])
            batch[j] = batch[j] + diff * [ignore_index]
        batches.append(torch.tensor(batch))
    return batches


def run_tagger(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)                
    l2ix = {'O':0,'B-Claim':1,'I-Claim':2}
    ix2l = {0:'O',1:'B-Claim',2:'I-Claim'}
    tst_sentences,tst_labels = read_gen_data(args.file_dir,l2ix)
    tst_featurized_sentences = get_data(tst_sentences,tokenizer,args.max_len)
    tst_data = DataLoader(dataset=tst_featurized_sentences,batch_size=args.bs,collate_fn=collate_fn)
    tst_data_y = batch_labels(tst_labels,args.bs)
    if args.is_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #device = torch.device("cuda")

    model = SequenceTagger(args.bert_dim,len(l2ix),args.bert_model_name)        
    model.load_state_dict(torch.load(args.load,map_location=device)["model_state_dict"])
    
    model.to(device)
    predictions = predict(model,tst_data,tst_data_y,args.is_gpu)    
    p2 = [token for sentence in predictions for token in sentence]
    g2 = [token for sentence in tst_labels for token in sentence]
    #pdb.set_trace()
    precision, recall, fscore, support = score(g2,p2)
    if args.verbose:
        report_performance(-1,-1,precision,recall,fscore,p2,g2)

    # If you  do not want to see claim spans with only 3-4 tokens set cover_sentence to True
    if args.cover_sentence:        
        _predictions2 = [[ ix2l[token_pred] for token_pred in sentence_pred ]for sentence_pred in predictions]
        predictions2 = []
        for __preds in _predictions2:        
            if len(list(set(__preds))) >= 2:
                __preds_new = ["B-Claim"]+ ["I-Claim" for i in range(len(__preds)-1)]
                predictions2.append(__preds_new)
            else:
                predictions2.append(__preds)                    
    else:
        predictions2 = [[ ix2l[token_pred] for token_pred in sentence_pred ]for sentence_pred in predictions]
        
    with open(args.out_file_name, 'w') as f:
        json.dump(predictions2, f)

    if args.prediction_conll_output:
        write_to_file(args.tagger_predictions_conllu_fname,predictions2,tst_sentences)
    return predictions2



def main():
    parser = argparse.ArgumentParser(description='Refactor code output with json file')
    parser.add_argument('--file_dir',default='/projekte/mardy/eday/refactor/moducles/claim_identifier/bilstm_cnn_crf_tagger/data/MARDY/mardy_political_enriched/Sentence_Level/test.dat.abs',help='input file')
    parser.add_argument("--max_len",type=int,default=256)
    parser.add_argument("--bs",type=int,default=16)
    parser.add_argument("--bert_dim",type=int,default=768)
    parser.add_argument("--bert_model_name",default='bert-base-german-cased')
    parser.add_argument("--is_gpu",type=str2bool, default=True,
                        help='gpu or cpu',choices=['True (Default)', True, 'False', False])    
    parser.add_argument('--predictions_conll_output', type=str2bool, default=True, help='create a prediction file', nargs='?',
                        choices=['True (Default)', True, 'False', False])
    parser.add_argument('--verbose', type=str2bool, default=True, help='create a prediction file', nargs='?',
                        choices=['True (Default)', True, 'False', False])
    parser.add_argument('--cover_sentence', type=str2bool, default=True, help='Extend small claim spans to whole sentence', nargs='?',
                        choices=['True (Default)', True, 'False', False])    
    parser.add_argument('--tagger_predictions_conllu_fname', '-s', default='predictions.abs',help='name of prediction file.')
    parser.add_argument('--load',type=str,default='2019_05_07_16-47_03_model.bin',help='path of the model used for generation')
    parser.add_argument('--prediction_conll_output', type=str2bool,
                        default=True, help='Create prediction file in conllu format.',nargs='?',
                        choices=['yes (Default)', True, 'no', False])    
    
    
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)                

    l2ix = {'O':0,'B-Claim':1,'I-Claim':2}
    ix2l = {0:'O',1:'B-Claim',2:'I-Claim'}
    tst_sentences,tst_labels = read_gen_data(args.file_dir,l2ix)
    tst_featurized_sentences = get_data(tst_sentences,tokenizer,args.max_len)
    tst_data = DataLoader(dataset=tst_featurized_sentences,batch_size=args.bs,collate_fn=collate_fn)
    tst_data_y = batch_labels(tst_labels,args.bs)


    if args.is_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SequenceTagger(args.bert_dim,len(l2ix),args.bert_model_name)        
    model.load_state_dict(torch.load(args.load,map_location=device)["model_state_dict"])
    model.to(device)
    predictions = predict(model,tst_data,tst_data_y,args.is_gpu)
    p2 = [token for sentence in predictions for token in sentence]
    g2 = [token for sentence in tst_labels for token in sentence]
    precision, recall, fscore, support = score(g2,p2)

    if args.verbose:
        report_performance(-1,-1,precision,recall,fscore,p2,g2)
    # If you  do not want to see claim spans with only 3-4 tokens set cover_sentence to True
    if args.cover_sentence:        
        _predictions2 = [[ ix2l[token_pred] for token_pred in sentence_pred ]for sentence_pred in predictions]
        predictions2 = []
        for __preds in _predictions2:        
            if len(list(set(__preds))) >= 2:
                __preds_new = ["B-Claim"]+ ["I-Claim" for i in range(len(__preds)-1)]
                predictions2.append(__preds_new)
            else:
                predictions2.append(__preds)                    
    else:
        predictions2 = [[ ix2l[token_pred] for token_pred in sentence_pred ]for sentence_pred in predictions]
        
    #with open(args.out_file_name, 'w') as f:
    #    json.dump(predictions2, f)

    if args.prediction_conll_output:
        write_to_file(args.tagger_predictions_conllu_fname,predictions2,tst_sentences)
        

    return predictions 
        
def predict(model,datax,datay,is_gpu):
    model.eval()
    predictions = [] 
    for x_batch,y_batch in zip(datax,datay):
        logits = model(x_batch,y_batch,is_eval=True,is_gpu=is_gpu)
        values,indices = torch.max(logits,2)
        tmp = y_batch == -1
        tmp = tmp.sort(dim=1)[0]
        first_nonzero = (tmp == 0).sum(dim=1)
        for i in range(len(first_nonzero)):
            predictions.append(indices[i,:first_nonzero[i]].tolist())
    return predictions 

def write_to_file(out_name,predictions,datax,gold_labels=None):
    with open(out_name, 'w') as f:
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                pred = predictions[i][j]
                word = datax[i][j]
                if gold_labels == None:
                    line = "{}\t{}\n".format(word,pred)
                else:
                    true_label = gold_labels[i][j]
                    line = "{}\t{}\t{}\n".format(word,pred,true_label)
                f.write(line)
            f.write("\n")

if __name__ == '__main__':
    main()
