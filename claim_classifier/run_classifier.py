import pdb
from tqdm import tqdm, trange
import sys
import numpy as np
import os,random
import math
import time,datetime
import argparse
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from data import *
from multi_label_model import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from pytorch_transformers import *


from sklearn.metrics import roc_curve, auc,f1_score,recall_score,precision_score
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
#n_gpu = torch.cuda.device_count()
n_gpu = 1

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if n_gpu > 0:
    torch.cuda.manual_seed_all(42)




def get_datetime_str():
    d = datetime.datetime.now()
    return '%02d_%02d_%02d_%02d-%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute, d.second)

def numpy_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def f1_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = f1_score(golds[:,i],preds[:,i])
    #TODO:: Add overall micro/macro f1 calculation 
    #f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1

def precision_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = precision_score(golds[:,i],preds[:,i])
    #TODO:: Add overall micro/macro f1 calculation 
    #f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1

def recall_measures(preds,golds):
    f1 = dict()    
    for i in range(golds.shape[1]):
        f1[i] = recall_score(golds[:,i],preds[:,i])
    #TODO:: Add overall micro/macro f1 calculation 
    #f1["micro"] = f1_score(golds.ravel(),preds.ravel())
    return f1

def metrics(preds,golds):
    res = np.equal(preds,golds)
    acc = np.mean(res,axis=1).sum()/golds.shape[0]
    f1  = f1_measures(preds,golds)
    p1 = precision_measures(preds,golds)
    r1 = recall_measures(preds,golds)
    for u in f1.keys():
        f1[u] = " {:.3f}".format(f1[u])
        p1[u] = " {:.3f}".format(p1[u])
        r1[u] = " {:.3f}".format(r1[u])
    result = {'accuracy': "{:.3f}".format(acc),
              'f1': f1,
              'precision':p1,
              'recall':r1}
    return result
    
def main_loop(model,optimizer,schedular,train_dataloader,eval_dataloader,args,logger,num_epocs=1):
    global_step = 0
    model.train()
    best_micro_auc = -1 
    for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            schedular.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        logger.info('Eval after epoc {}'.format(i_+1))
        result = evaluate(model,eval_dataloader,args,logger)
        if result['roc_micro'] > best_micro_auc:
            best_micro_auc = result['roc_micro']            
            print("saved model at {} epoch with {} micro-auc value".format(i_+1,best_micro_auc))
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), args["output_model_file"])
            # print info and model saved message


def predict(model, path, args,tokenizer,test_filename='test.tsv'):
    predict_processor = MardyMultiLabelTextProcessor(path)
    test_examples = predict_processor.get_test_examples(path, test_filename, size=-1)    
    # Hold input data for returning it 
    input_data = [{ 'id': input_example.guid, 'comment_text': input_example.text_a } for input_example in test_examples]

    test_features = convert_examples_to_features(
        test_examples, args["label_list"], args['max_seq_length'], tokenizer)
    
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    
    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

    # RUN EVALUATION:
    all_logits = None
    
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    print("## INSIDE PREDICT FUNCTION ### ")
    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
            
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=args["label_list"]), left_index=True, right_index=True)

def evaluate(model,eval_dataloader,args,logger,is_final=False):    
    all_logits = None
    all_labels = None    
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
            
        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:    
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        #print("eval example size:",input_ids.size(0))
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
#     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(args["num_labels"]):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    #print(all_logits)
    #print(all_logits.shape)
    f1_preds= 1 * (numpy_sigmoid(all_logits) > 0.5)
    f1 = metrics(f1_preds,all_labels)

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
#               'loss': tr_loss/nb_tr_steps,
              'f1': f1,
              'roc_micro':roc_auc['micro']}
    
    output_eval_file = args['output_dir']    
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results *****")
        if is_final == True:
            writer.write("\n ----- Best model on test set ----- \n")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        writer.write("\n ----------------------------------------------------\n")
    return result

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()

def get_predictions_from_out_file(fname_preds,fname_golds,thresh=0.5):
    df = pd.read_csv(fname_preds)
    result_mx = df.loc[:,'100':'900'].as_matrix()
    preds = 1* (result_mx>thresh)
    df_gold = pd.read_table(fname_golds)
    result_mx_gold_ =  [ x.split(' ') for x in df_gold.loc[:,'major_classes'].tolist()]
    result_mx_gold = np.array(result_mx_gold_)
    a1  = result_mx_gold.astype(int)
    a2  = preds.astype(int)
    res = np.equal(a1,a2)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True                  
    
def main():
    model_state_dict = None
    exp_date = get_datetime_str()
    parser = argparse.ArgumentParser(description='Run trained classifier from the checkpoint file')
    parser.add_argument('--train_size', type=int, default=-1),
    parser.add_argument('--val_size', type=int, default=-1),
    parser.add_argument('--full_data_dir',
                        default='./data/'),
    parser.add_argument('--task_name',
                        default='mardy_multilabel')    
    parser.add_argument('--data_dir',
                        default='./data/tmp')
    parser.add_argument('--do_lower_case', type=str2bool, default=False,
                        help='False to continue training the word embeddings.', nargs='?',
                        choices=['yes', True, 'no (default)', False]),
    parser.add_argument('--output_dir',
                        default='./saved_models/{}_report.log'.format(exp_date)),
    parser.add_argument('--file_dir',
                        default='./data/test.tsv'.format(exp_date)),    
    parser.add_argument('--output_fname',
                        default='./classifier_predictions.csv'.format(exp_date)),
    
    parser.add_argument('--bert_model',
                        default='bert-base-german-cased'.format(exp_date)),
    parser.add_argument('--model_file',
                        default='./saved_models/claim_classifier_model.bin'.format(exp_date)),    
    parser.add_argument('--max_seq_length', type=int, default=100),
    parser.add_argument('--do_train', type=str2bool, default=False,
                        choices=['yes', True, 'no (default)', False]),
    parser.add_argument('--do_eval', type=str2bool, default=False,
                        choices=['yes', True, 'no (default)', False]),
    parser.add_argument('--eval_batch_size', type=int, default=32),
    parser.add_argument('--train_batch_size', type=int, default=32),
    parser.add_argument('--learning_rate', type=float, default=5e-5),
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training."),
    parser.add_argument("--num_train_epochs",type=int,default=7),
    parser.add_argument("--hs",type=int,default=768),
    parser.add_argument("--seed",type=int,default=42),
    parser.add_argument("--local_rank",type=int,default=-1),
    parser.add_argument('--dp', type=float, default=0.1),
    parser.add_argument('--no_cuda', type=str2bool, default=False, nargs='?',
                        choices=['True', True, 'False (Default)', False])
    _args = parser.parse_args()
    args = vars(_args)


    processors = {
        "mardy_multilabel": MardyMultiLabelTextProcessor
    }
    task_name = args['task_name'].lower()
    processor = processors[task_name](args['data_dir'])
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args["label_list"] = label_list
    args["num_labels"] = num_labels

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args['model_file'])
    model = BertForMultiLabelSequenceClassification(args)
    model.load_state_dict(model_state_dict)
    model.to(device)


    tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])
    # Prediction 
    result = predict(model, args['full_data_dir'],args,tokenizer,test_filename=args['file_dir'].split('/')[-1])
    result.to_csv(args['output_fname'], index=None)

if __name__ == '__main__':
    main()

