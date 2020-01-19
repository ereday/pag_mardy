import time,datetime
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

def get_datetime_str():
    d = datetime.datetime.now()
    return '%02d_%02d_%02d_%02d-%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute, d.second)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True                  


def print_cm(cm, labels):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            print(cell, end=" ")
        print()
    
def report_performance(epoch,lss,precision,recall,fscore,preds,golds):
    print("epoch:{}\ttrn_lss:{:0.3f}".format(epoch,lss/len(golds)))
    print("")

    line_p  = "p:{:0.3f}\t{:0.3f}\t{:0.3f}".format(precision[0],precision[1],precision[2])
    line_r  = "r:{:0.3f}\t{:0.3f}\t{:0.3f}".format(recall[0],recall[1],recall[2])
    line_f1 = "f:{:0.3f}\t{:0.3f}\t{:0.3f}".format(fscore[0],fscore[1],fscore[2])
    print(line_p)
    print(line_r)
    print(line_f1)
    print("\nconfusion matrix:")
    label_names= [ str(a) for a in list(set(golds))]
    print_cm(confusion_matrix(golds,preds,labels=list(set(golds))),label_names)
    print("----------------------------------------------------")
    


def get_report_header(score_names):
    text = 'Evaluation\n'
    header = '\n\n %15s |' % 'epoch '
    for n, score_name in enumerate(score_names):
        header += ' %17s ' % score_name
        if n < len(score_names) - 1:
            header += '|'

    text += header
    blank_line = '\n' + '-' * len(header)
    text += blank_line
    return text




def write_epoch_scores(text, epoch, _scores):
    scores = [[_scores[i][j] for i in range(len(_scores))] for j in range(len(_scores[0]))]
    text += '\n %15s |' % ('%d'% epoch)
    for n, score in enumerate(scores):
        text +=' %5s,%5s,%5s ' % ('{:1.2f}'.format(score[0]),'{:1.2f}'.format(score[1]),'{:1.2f}'.format(score[2]))
        #text += ' %14s ' % ('%1.2f' % score)
        if n < len(scores) - 1:
            text += '|'
    return text


    
