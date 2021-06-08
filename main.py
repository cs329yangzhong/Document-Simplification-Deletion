import torch
import torch.nn as nn
from torch import optim
import time
import os
import argparse
import utils
import numpy as np
from ast import literal_eval
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc,average_precision_score, precision_recall_curve
# from train import *
from model.CNN_glove import CNN_Glove
from model.FCN_glove import FCN_Glove
from utils import *
from data_loading import Corpus, DataProducer_glove, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from model.LR import LR 
from model.BiLSTM_Glove import BiLSTM_glove
from model.FCN import FCN

if __name__ == "__main__":
                
    print(torch.cuda.current_device())
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(123)
    #np.random.seed(0)
    parser = argparse.ArgumentParser(description='Sentence classificer')
    parser.add_argument('-Global', type=bool, default=False)
    parser.add_argument('-elmo', type=bool, default=False, help="whether or not use ELMO embedding")
    # learning
    parser.add_argument("-concat_surround", type=bool, default=False)
    parser.add_argument('-add_features', type=bool, default=False, help="whether adding features")
    parser.add_argument('--levels', type=int, default=4,
                        help='# of levels (default: 4)')
    parser.add_argument('--nhid', type=int, default=150,
                        help='number of hidden units per layer (default: 150)')
    parser.add_argument('-training', type=str, default='training_final_ordered.txt', help='Training dataset ')
    parser.add_argument("-binning", type=int, default=None, help='binning number')
    parser.add_argument("-abl", type=str, default='None')
    parser.add_argument('-path', type=str, default='./data/', help='Path of datasets')
    parser.add_argument('-corpus', type=str, default='training_featured.txt', help='Name of corpus file')
    parser.add_argument('-model', type=str, default="Self_attention", help='Model choose from Logstics Regression, CNN , BiLSTM and transferLearning')
    parser.add_argument('-lr', type=float, default=0.05, help='initial learning rate [default: 0.0005]')
    parser.add_argument('-non_linear', type=bool, default=True, help="non_linearlity for concatnating features")
    parser.add_argument('-epochs', type=int, default=30, help='number of epochs for train [default: 256]')
    parser.add_argument('-batch_size', type=int, default= 16, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=150, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
    parser.add_argument("-class_to_train", type=str, default="G4", help="which class to train [G7, G4]")
    parser.add_argument("-plot", type=bool, default=True, help="whether to save the loss in a plot")
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-numClass', type=int, default="1", help="number of class for classification")
    parser.add_argument('-hidden_dim', type=int, default = 150, help="number of hidden dimention")
    parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel_num', type=int, default=300, help='number of each kind of kernel')
    parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
    # device
    parser.add_argument("-delete", type=bool, default=True, help="the model is checking deletion")
    parser.add_argument("-use_gpu", type=bool, default=False, help="whether use gpu")
    # parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument("-conc", type=int, default=0, help="concactanation of addtional features")

    args = parser.parse_args()

    # make logger.
    model_name = args.model
    logger = utils.get_logger(model_name, args)

    logger.info('Arguments: {}'.format(args))

    if torch.cuda.current_device() != -1:
        args.use_gpu = True

    n_gpu = torch.cuda.device_count()

    if args.plot:
        all_train_loss = []
        all_train_acc = []
        all_step = []
        all_valid_loss = []
        all_valid_acc = []
   
    # Setting for weigted class.
    if args.class_to_train == "G4":
        weight_ = 10.0364
    if args.class_to_train == 'G7':
        weight_ = 3.8073
    
    ## feature modification.
    
        # get the features after ablation.
    NE = ['ORGANIZATION', 'PERCENT', 'PERSON', 'DATE', 'MONEY', 'TIME', 'LOCATION']
    Surface = ['num_number', 'ari', 'stopword', 'sent_length'] 
    Pos = ["doc_pos", "parag_pos", "parag_relative_pos"]
    Global = ['sent_num','token_num','Arts','Kids','Health','Science','Money','Law','War & Peace','Sports']
    Discourse = ['Comparison',  'Temporal','Contingency' , 'Expansion', 'discourse_S', 'discoures_M']
    feature_map = { "Pos": Pos, "Discourse": Discourse, "Global":Global}
    features = []
        
    Global_tag = args.Global 
    features = []
    for item in feature_map:
        if item != args.abl:
            features += feature_map[item]
    args.conc = len(features)

    if not args.add_features:
        args.conc = 0

    print("concat feature number ", args.conc)
    print(args.model)

    if args.model == "CNN_Glove" or args.model== 'BiLSTM_Glove' or args.model == 'StackLSTM' or args.model == "FullNN" or args.model == "ElmoCNN" or args.model == "FCN_Glove":
        print("GLOVE models")
        start_time = time.time()
        print(args.add_features)
        if args.class_to_train == "G4":
            weight_ = 1.5
        if args.class_to_train == 'G7':
            weight_ = 4
        weight_ = 1
        if args.add_features == False:
            args.conc = 0
        embedding_file = "glove.42B.300d.txt"
        model_corpus = Corpus()
        train_set = DataProducer_glove(args.path, args.training, model_corpus, args= args, elmo=args.elmo, class_to_train=args.class_to_train, abl=args.abl)
        valid_set = DataProducer_glove(args.path, "valid", model_corpus,args= args,class_to_train=args.class_to_train, elmo=args.elmo, abl=args.abl)
        test_set = DataProducer_glove(args.path,"test", model_corpus,args= args, class_to_train=args.class_to_train,abl=args.abl, elmo=args.elmo)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)

        pos_weight_ = torch.FloatTensor([weight_]).cuda()
        best_model = None
        best_val_f1 = 0  # Need to apply early stopping.
        best_val_loss = 1000
        tol = 0


        for lr in [2e-5]:
            best_val_f1 = 0
            args.lr = lr
            lr = lr
            #args.add_features = False
            if args.add_features == False:
                args.conc = 0
            print(args.conc)
            logger.info("lr @ %s"%lr)
#
            tol = 0
            
            # 3 runs for each.
            p_r_f = []
            best_roc_auc = 0
            #for mlp in [50, 100, 150, 200]:
            for batch_size in [32, 64]:
                best_val_f1 = 0
                #args.hidden_dim = mlp
                all_train_loss = []
                all_valid_loss = []
                all_valid_acc = []


                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                valid_loader = DataLoader(valid_set, batch_size=batch_size)
                test_loader = DataLoader(test_set, batch_size=batch_size)

                logger.info("add feature %s"%args.add_features)
                if args.model == "CNN_Glove":
                    model = CNN_Glove(args, model_corpus.word_idx, embedding_file, encoding="utf-8")
                elif args.model == "BiLSTM_Glove":
                    model = BiLSTM_glove(args, model_corpus.word_idx, embedding_file, conc=args.concat_surround, encoding = 'utf-8')
                elif args.model == "FullNN":
                    model = FCN(args,model_corpus.word_idx, embedding_file,concat=args.concat_surround, Gaussian_num=args.binning)
                elif args.model == "LR_Glove":
                    model = LR(args, model_corpus.word_idx, embedding_file, encoding="utf-8")
                elif args.model == "FCN_Glove":
                    model = FCN_Glove(args, model_corpus.word_idx, embedding_file, mlp_d=150, encoding="utf-8",concat=args.concat_surround)
                
                # Training
                print(model)
                model = model.cuda()
                criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight_)
                optimizer = optim.Adam(model.parameters(), lr=args.lr)  
                tol = 0
                for epoch in range(args.epochs):
                    avg_loss = 0.0
                    truth_res = []
                    pred_res = []
                    count = 0
                    time1 = time.time()
                    batch_total = len(train_loader)
    		
                    # training
                    model.train()
                    for idx, training_data in enumerate(train_loader):
                        
                        train_inputs, train_len, train_feature, train_labels = training_data
                         
                        train_labels = train_labels.view(-1)
                        model.batch_size = len(train_labels)
    
                        ### FORWARD AND BACKWARD
                        output = model(train_inputs, train_feature, l1=train_len, batch_size=model.batch_size)
                        loss = criterion(output, train_labels)
                        optimizer.zero_grad()

                        loss.backward()
                        optimizer.step()
                        m = nn.Sigmoid()
                        pred_res += list(torch.round(m(output)).cpu().detach().numpy())
                        truth_res += [y for y in list(train_labels.cpu().detach().numpy())]    
                        avg_loss += loss.item()
                        count += 1
    
                        if idx % 50 == 0:
    #                        print('Epoch: %3d/ %3d'%((epoch+1),args.epochs
                            print (f'Epoch: {epoch+1:03d}/{args.epochs:03d} | ' f'Batch {idx:03d}/{len(train_loader):03d} | '
                                                                    f'Cost: {loss:.4f}')
                    print("for this epoch, we use %s s" % (time.time() - time1))
                    avg_loss /= (len(train_loader))
                    acc, f1_metrics = get_accuracy(truth_res, pred_res)
    
                    all_train_loss.append(avg_loss)
                    all_train_acc.append(f1_metrics[2])
    
                    logger.info("Epoch %s in %s ===> Training loss: %s   Acc %.4f f1_score %.4f      f1_metrics %s" % (
                    epoch + 1, args.epochs, avg_loss, acc, f1_metrics[2], str(f1_metrics)))
    
                    # validation
                    avg_loss = 0.0
                    truth_res = []
                    pred_res = []
                    count = 0
                    pred_prob = []
                    batch_total = len(valid_loader) * args.batch_size

                    acc, f1_metrics, loss = compute_binary_accuracy_f1(model, valid_loader,logger)
                    all_valid_loss.append(loss)
                    all_valid_acc.append(f1_metrics[2])
                    logger.info("Epoch %s in %s ===> Valid loss: %s , Acc %.4f : f1_score %s" % (epoch + 1, args.epochs, loss,  acc, str(f1_metrics)))
    
                    # Applying early stopping.
                    if  best_val_f1 < f1_metrics[2]:
                        best_model = model
                        #with open("runs/model_%s@lr_%s_%s_run%s_abl_%s.pt"%(args.model, lr,args.class_to_train,i, args.abl), 'wb') as f:
                         #   print('Save model!\n')
#                            torch.save(model, f)
    
                        best_val_f1 = f1_metrics[2]
                        best_tol = 0
                        print("Update....")
                        logger.info("Update Model with better val f1_score %.4f" % best_val_f1)
                    else:
                        tol += 1
                    if tol >= 60:
                        break
                fig, ax = plt.subplots(2, 1, figsize=(8, 12))
                ax[0].plot(range(len(all_valid_loss)), all_valid_loss, label='Validation loss')
                ax[0].plot(range(len(all_train_loss)), all_train_loss, label='Training loss')
                ax[0].set_xlabel('Epoch')
                ax[0].set_ylabel('Loss')
                ax[0].legend(loc='upper right')
                ax[1].plot(range(len(all_valid_loss)), all_valid_acc)
                ax[1].set_xlabel('Epoch')
                ax[1].set_ylabel('Accuracy')
                
                if args.binning == None:
                    bin_num = 0
                else:
                    bin_num = args.binning
                plt.savefig("plots/@lr-%s_plot_%s_%s_ablFeatures_%s_With_Surround%s-bin@%s-batch%s.jpg"%(lr, args.model, args.class_to_train, args.abl,args.concat_surround, args.binning, batch_size))
                plt.close()
    
                model = best_model
                ### Test

                avg_loss = 0.0
                truth_res = []
                pred_res = []
                pred_prob = []
                count = 0
                acc, f1_metrics, loss  = compute_binary_accuracy_f1(model, test_loader,logger)
                p_r_f.append([f1_metrics[0], f1_metrics[1], f1_metrics[2]])

                logger.info("Epoch %s in %s ===> With batch_size %s Test loss: %s , Acc %.4f : f1_score %s" % (epoch + 1, args.epochs,batch_size,  loss, acc, str(f1_metrics)))
               # for idx, valid_data in enumerate(test_loader):
               #     model.eval()
               #     valid_inputs, valid_len, valid_feature, valid_labels = valid_data
    
               #     valid_labels = valid_labels.view(-1)
               #     model.batch_size = len(valid_labels)
    
               #     # model.zero_grad()
               #     output = model(valid_inputs, valid_feature, l1=valid_len, batch_size=args.batch_size)
               #     loss = criterion(output, valid_labels)
               #     pred_prob += list(m(output).cpu().detach().numpy())
               #     
               #     m = nn.Sigmoid()
               #     pred_res += list(torch.round(m(output)).cpu().detach().numpy())
               #     truth_res += [y for y in list(valid_labels.cpu().detach().numpy())]
               #     avg_loss += loss.item()
               #     count += 1
               #     
               # precision, recall, thresholds = precision_recall_curve(truth_res, pred_prob)
               # fpr, tpr, threshold =  roc_curve(truth_res, pred_prob)
               # roc_auc = auc(fpr, tpr)
               # auc_ = auc(recall, precision)
               # logger.info("auc for pr is %s"%auc_)
               # logger.info("auc for roc is %s"%roc_auc)
               # macro = precision_recall_fscore_support(truth_res, pred_res, average="macro")
               # logger.info("overal f1_score%s"%str(macro))
               # ap = average_precision_score(truth_res, pred_res)
               # logger.info("average precision is %s"%ap)
               # print("for this epoch, we use %s s" % (time.time() - time1))
               # avg_loss /= (batch_total)

               # acc, f1_metrics = get_accuracy(truth_res, pred_res)
               # if f1_metrics[0] != 0:
               #     p_r_f.append([f1_metrics[0], f1_metrics[1], f1_metrics[2]])

               # logger.info(
               #     "the Test acc %s , AUC %s , and f1_metrics is %s" % (acc,auc_, str(f1_metrics)))
            average = np.array(p_r_f)
            logger.info("average f1 measure is %s and std is %s"%(np.mean(average,axis=0),np.std(average, axis=1)))
    
##        logger.info(
 #           "the average f1_metrics are %.4f %.4f %.4f" % (np.mean(pre), np.mean(rec), np.mean(f_score)))     
	
    # Self_attention
