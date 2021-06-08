import torch
import random
random.seed(30)
from random import sample
from sklearn.utils import shuffle
import numpy as np
import os 
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
from ast import literal_eval
#from bert_serving.client import BertClient
import time
import re
from sklearn.utils import resample
from nltk.tokenize import word_tokenize


article_2_len = None



def get_batch(batch_id, class_name, article_dic = None, delete=True,  file_path="./data/train_batch/"):
    if class_name == "G4":
        class_name = "G4is_summary"
        
    df1 = pd.read_csv((file_path+"batch_%s_label.tsv"%batch_id), sep="\t")
    if delete:
        label = [ 1 if y == 0 else 0 for y in df1[class_name].tolist()]
    elif (delete == False):
        label = df1[class_name]
    embed = np.load(file_path+"batch_%s_embed.npy"%batch_id)
    
    # get additional features.
    doc_id =list(df1["doc"])
    index = list(df1["index"])

    pos_lst = []
    #for i in range(len(doc_id)):
    #    pos_lst.append(getPosition(index[i],str(doc_id[i]), article_dic))

    #all_feature = df1.loc[:, "ORGANIZATION":]
    all_feature = []
    return embed, all_feature, label

def get_len_article():
    article_2_len = {}

    with open("./data/training.txt") as f:
        f = f.readlines()[1:]
        for line in f:
            line = line.split("\t")
            if line[0] not in article_2_len:
                article_2_len[line[0]] = 1
            else:
                article_2_len[line[0]] += 1

    with open("./data/testdata.txt") as f1:
        f = f1.readlines()[1:]
        for line in f:
            line = line.split("\t")
            if line[0] not in article_2_len:
                article_2_len[line[0]] = 1
            else:
                article_2_len[line[0]] += 1

    print(article_2_len)
    return (article_2_len)

#def getPosition(index, doc_id, article_2_len):
    # This helper function returns the percentile of the article.
    #article_2_len = get_len_article()
#    return (index+1) / article_2_len[doc_id]

def get_ValidOrTest(type, class_name, args, article_dic = article_2_len, delete=True, file_path="./data/valid_test_batch/"):
    if class_name == "G4":
        class_name = "G4is_summary"
    if type == 'V':
        file_name = "valid"
    elif type == "T":
        file_name = "test"
    df1 = pd.read_csv((file_path+"%s.tsv"%file_name), sep = "\t")
    if (delete): 
        label = [ 1 if y == 0 else 0 for y in df1[class_name].tolist()]
    elif (delete == False):
        label = df1[class_name]
    embed = np.load(file_path+"%s.npy"%file_name)

    doc_id = list(df1["doc"])
    index = list(df1["index"])
   
    pos_lst = []
    for i in range(len(doc_id)):
        pos_lst.append("None")
    list_ = []
    for i in range(len(label)//args.batch_size):
        label_batch = label[args.batch_size*i:args.batch_size * (i+1)]
        embed_batch = embed[args.batch_size*i:args.batch_size * (i+1)]
        pos_batch = pos_lst[args.batch_size*i:args.batch_size*(i+1)]
        list_.append((label_batch,pos_batch,  embed_batch))
    return list_


## Class for Logistic Regression.
class DataProducer(Dataset):
    def __init__(self, path, file_name, batch_size=128, max_len= 128, class_name="G7", cuda=False, encoding = 'utf-8'):

        """
        :param corpus:      Corpus contains all np.ndarray for the corresponding data. shape (total_seq, seq_num, embed_num)

        :param max_len:     Max length of a sent
        """
        df1 = pd.read_csv(path + file_name, sep="\t")
        print(len(df1))
        self.max_length = max_len
        self.cuda = cuda
        self.sents_vec = ([literal_eval(y) for y in list(df1["embedding"])])
        self.labels = [int(y) for y in list(df1[class_name])]

class DataProducer_sents(Dataset):

    def __init__(self, path, file_name, max_len=128, class_name="G7", cuda=False, encoding='utf-8'):
        """
        :param corpus:      Corpus contains all np.ndarray for the corresponding data. shape (total_seq, seq_num, embed_num)
        
        :param max_len:     Max length of a sent
        """
        df1 = pd.read_csv(path+file_name,sep = "\t")
        print(len(df1))
        self.max_length = max_len
        self.cuda = cuda
        self.sents_vec = ([literal_eval(y) for y in list(df1["embedding"])])
        self.labels = [int(y) for y in list(df1[class_name])]

    def __getitem__(self, index):
        """
        :return: Tensors of sent, sentence length, concatenating features and label(s)
        """
        seqs = self.sents_vec[index]
        sent = torch.FloatTensor(seqs)
        label = torch.FloatTensor([self.labels[index]])

        if self.cuda:
            sent, label = sent.cuda(),  label.cuda()
        return sent, label

    def __len__(self):
        return len(self.sents_vec)


# healer function to check label.
'''
parameters
    K -> keep
    D -> Deletion
    SD -> Start Of Deletion
    ED -> End Of Deletion
'''
def change_label(labels):
    for index in range(len(labels)):

        if index == 0:
            if labels[index] == 1:
                labels[0] = "K"
            else:
                if labels[1] == 1:
                    labels[0] = "SD"
                else:
                    labels[0] = "SD"

        elif index == (len(labels)-1):

            if labels[index] == 1:
                labels[index] = "K"
            else:

                if labels[index-1] == "K":
                    labels[index] = "D"

                else:
                    labels[index] = "D"

        # for the middle labels:
        else:
            cur_label = labels[index]
            if cur_label == 1:
                labels[index] = "K"
            else:
                if labels[index-1] == "K" and labels[index+1] == 1:
                    labels[index] = "D"
                elif labels[index-1] == "K" and labels[index+1] == 0:
                    labels[index] = "SD"
                elif labels[index-1] == "D" and labels[index+1] == 0:
                    labels[index] = "D"
                elif labels[index - 1] == "SD" and labels[index + 1] == 1:
                    labels[index] = "D"
                elif labels[index - 1] == "SD" and labels[index + 1] == 0:
                    labels[index] = "D"
                elif labels[index - 1] == "D" and labels[index + 1] == 1:
                    labels[index] = "D"
   # print(labels)
    return labels

# if __name__ == "__main__":
#     change_label(["1","0","1","0","0","0","1","1","0","0"])

class DataProducer_doc(Dataset):

    def __init__(self, path, file_name, max_len=75, delete =True, class_name="G7", cuda=True, encoding="utf-8"):

        self.data = pd.read_csv(path+file_name, sep="\t")

        self.max_length = max_len
        self.cuda = cuda
        self.delete = delete
        if (self.delete):
            self.labels = ["1" if y =="0" else "0" for y in self.data[class_name].tolist()]
        else:
            self.labels = [int(y) for y in list(self.data[class_name])]
        self.doc_list, self.label_list = self.create_docList(self.data, class_name)
        

    def create_docList(self, data, classname):
        # This is use to create the sequence for each document and prepared for BiLSTM-CRF.
        doc_group = []
        label_group = []
        doc_tag = -1
        for idx, row in self.data.iterrows():
            doc_id = row["doc"]
            if doc_id != doc_tag:
                # new doc
                if doc_tag == -1:
                    new_doc = []
                    new_tags = []
                    new_doc.append(literal_eval(row["embedding"]))
                    new_tags.append(row[classname])
                else:
                    doc_group.append(new_doc)
                    label_group.append(new_tags)
                    new_doc = []
                    new_tags = []
                    new_doc.append(literal_eval(row["embedding"]))
                    new_tags.append(row[classname])

                doc_tag = doc_id
            else:
                new_doc.append(literal_eval(row["embedding"]))
                # #print(type(row[classname]))
                new_tags.append(row[classname])

        new_lable_group = []
        for tags in label_group:
            
            tags = ["1" if y == 0 else "0" for y in tags]

            new_lable_group.append(tags)
        return doc_group, new_lable_group

       # print(len(doc_group))

    def __getitem__(self, index):
        max_length = 75

        lens_s = len(self.label_list[index])
        if lens_s > max_length:
            lens_s = 75


        sent_vec = self.doc_list[index]

        seq_len = len(sent_vec)

        if seq_len < max_length:
            padding = max_length - seq_len
            old = np.expand_dims(sent_vec, axis= 0)
            new_added = np.expand_dims(np.asarray([[0] * 1024] * padding), axis=0)
            result = np.concatenate((old,new_added), axis = 1)

            new_xs = np.squeeze(result, axis = 0)
        else:
            new_xs = np.asarray(sent_vec[:max_length])


        if len(self.label_list[index]) <= 75:
            padding = 75 - len(self.label_list[index])
            new_label = self.label_list[index] + (["<pad>"]*padding)
        else:
            new_label = self.label_list[index][:75]
        #print(new_label)
        return new_xs, new_label, lens_s

    def getIndex(self, index):
        return self.doc_list[index], self.label_list[index]


    def __len__(self):
        return len(self.doc_list)

##############################################################################
### For glove.
############3
#############
############
def word_2_ids(sents):
    """
    :param sents: A list of lists where each list corresponding to a sentence
    :return: A dictionary contains pairs of word and its index
    """
    vocab = []
    for sent in sents:
        for word in sent:
            vocab.append(word.lower())
    word_to_ix = {}
    word_to_ix["<s>"] = 0
    word_to_ix["</s>"] = 1
    for word in vocab:
        if word.lower() in word_to_ix:
            continue
        else:
            word_to_ix[word.lower()] = len(word_to_ix)

    return word_to_ix

def clean_str(string):
   
 #   print(string)
    # first exclude the extra quote.
    string = string[1:-1]
    string = string.replace("|", "")
    string = " ".join(word_tokenize(string))
    return string

# a class to store the vocabs used for word embedding
class Corpus:
    def __init__(self, path="data/", file_name="training_final_ordered.txt", encoding='utf-8'):
        df = pd.read_csv(path+file_name, sep= "\t")
        lines = df["sent"]
        print(lines)
        lines = [[y.lower() for y in clean_str(line).split()] for line in lines]
        lines.sort(key=lambda x: len(x), reverse=True)

        self.sents = lines
        self.word_idx = word_2_ids(self.sents)

# preparing data to be loaded via torch.utils.data.Dataloader
class DataProducer_glove(Dataset):

    def __init__(self, path, file_name, corpus, class_to_train, args, abl=None, avg_label=True, round_up=False, max_len=60, cuda=True, encoding='utf-8', elmo=False, concat_surround=False):
        """
        :param corpus:      Corpus to be used for indexing
        :param
        :param max_len:     Max length of a sent
        """

        # get the features after ablation.
        NE = ['ORGANIZATION', 'PERCENT', 'PERSON', 'DATE', 'MONEY', 'TIME', 'LOCATION']
        Surface = ['num_number', 'ari', 'stopword',"sent_length"]
        Pos = ["doc_pos", "parag_pos", "parag_relative_pos"]
        Global = ['sent_num', 'token_num', 'Arts', 'Kids', 'Health', 'Science', 'Money', 'Law', 'War & Peace', 'Sports', 'fkg', 'fre', 'smog', 'cli', 'dcrs', 'lwr', 'gf', 'ari']
        Discourse = ['Comparison', 'Temporal', 'Contingency' ,'Expansion', 'discourse_S', 'discoures_M', 'depth','nucleus','root', 'elabration', 'attribution', 'joint', 'contrast', 'evaluation', 'explanation', 'same-unit'] + ["previous_%s" % i for i in range(1, 101)] + ["after_%s" % i for i in range(1, 101)]
        #Discourse = ['Comparison',  'Temporal','Contingency' , 'Expansion', 'discourse_S', 'discoures_M']                                                                                         
        feature_map = {"Pos": Pos,"Global": Global, "Discourse": Discourse}
        features = []
        feature_map = {"Pos": Pos}
        for item in feature_map:
            if item != args.abl:
                features += feature_map[item]

        #if args.concat_surround:
         #   features = ["previous_%s" % i for i in range(1, 101)] + ["after_%s" % i for i in range(1, 101)]
          #  print("Concat before and after sentences")
        args.conc = len(features)

        if class_to_train == "G4":
            class_to_train = "G4is_summary"

        test_valid = pd.read_csv(path + "chao_test_ordered.txt", sep="\t")
        valid_docs = [' marktwain-newspaper.en', ' dali-vr.en', ' nativeamerican-diets.en', ' airstrikes-iraq.en', ' syria-inspectors.en', ' periodictable-elements.en', ' china-aviation.en', ' comcast-merger.en', ' amtrak-crash.en', ' eggprices-rising.en', ' drones-wildfires.en', ' asteroid-corral.en', ' aquaponics-farm.en', ' harvarddebate$versusinmates.en', ' 3d-indoormap.en'] 
        test_docs = [' timetravel-paradox.en', ' sesamestreet-preschool.en', ' solar-panels.en', ' asian-modelminority.en', ' concussion-stud$.en', ' auschwitz-palestine.en', ' bee-deaths.en', ' botany-students.en', ' doodler-nebraskalawmaker.en', ' shuttle-parts.en', ' alienplanet-swim.en', ' emergen$yresponse-robots.en', ' miami-searise.en', ' migrantkids-uprooted.en', ' deportee-videogame.en', ' dinosaur-colors.en', ' digital-giving.en', ' hawaii-homeless.$n', ' libya-boatcapsize.en', ' google-selfcars.en', ' iran-water.en', ' return-trip.en', ' asia-ozone.en', ' pakistan-earthquake.en', ' chinook-recognition.en', ' dog-drinking.en', ' school-threats.en', ' muslim-challenges.en', ' syria-refugees.en', ' football-virtualreality.en', ' military-police.en', ' class-sizes.en',' antelopevalley-bomber.en', ' basketball-mentors.en', ' koala-trees.en']

        if file_name == "valid":
            df = test_valid[test_valid['doc'].isin(valid_docs)]
            df = shuffle(df, random_state=0).reset_index(drop=True)
            df = df[~df['sent'].str.contains("## ")]
            df = df[~df['sent'].str.contains("<img")]

            valid = df[['sent', "G4is_summary", "G7"]]
            valid.to_csv("data/valid.tsv", sep="\t")
        elif file_name == "test":
            df = test_valid[test_valid['doc'].isin(test_docs)]
            df = shuffle(df, random_state=0).reset_index(drop=True)
            df = df[~df['sent'].str.contains("## ")]
            df = df[~df['sent'].str.contains("<img")]
            
            test = df[['sent', "G4is_summary", "G7"]]
            test.to_csv("data/test.tsv", sep="\t")
        else:
          #  file_name = "testdata_all_featured.txt_"
            df = pd.read_csv(path + file_name, sep="\t")
            df = df[df['sent'].str.len() > 3]
            # df = df[df.depth != -1]
            df = df.dropna()


            if class_to_train in ['G7', "G4is_summary"]:
                df_majority = df[df[class_to_train] == 1]
                df_minority = df[df[class_to_train] == 0]

            else:
                df_majority = df[df[class_to_train] == 0]
                df_minority = df[df[class_to_train] == 1]
            df_majority_downsampled = resample(df_majority,
                                               replace=False,  # sample without replacement
                                               n_samples=len(df_minority),  # to match minority class
                                               random_state=123)  # reproducible results

            # Combine minority class with downsampled majority class
            df_downsampled = pd.concat([df_majority_downsampled, df_minority])
            df = df_downsampled

            # save data.
            training = df[["sent", class_to_train]]
            training.to_csv("data/train.tsv", sep="\t")

            print(df.shape)
        
        self.sents = [clean_str(y).split() for y in df["sent"]]
        #self.sents = [ y for y in self.sents if len(y) >= 3]

        # only include previous and after sentence features.
        # features = []
        self.features = df.loc[:, features] 
        
        if class_to_train in ["G7", "G4is_summary"]: 
            self.labels = [1 if int(y) == 0 else 0 for y in df[class_to_train]]
        else:
            self.labels = [0 if int(y) == 0 else 1 for y in df[class_to_train]]
            
        self.elmo = elmo
        # Truncate the dataset to well-fit in the batch.
        #if file_name not in ["valid", "test"]:
        #    max_batch = (len(self.sents) -(len(self.sents)% args.batch_size)) // args.batch_size 
        #    batch_size = args.batch_size
        #else:
        #    max_batch = len(self.sents)// args.batch_size
        #    batch_size = args.batch_size
        #self.sents = self.sents[:max_batch * batch_size]
        #self.labels = self.labels[:max_batch * batch_size]
        #self.features = self.features.iloc[:max_batch*batch_size, :]
        assert len(self.sents) == self.features.shape[0]

        self.word_to_ix = corpus.word_idx
        self.max_length = max_len
        self.avg_label = avg_label
        self.round = round_up
        self.cuda = cuda

    def __getitem__(self, index):
        """
        :return: Tensors of sent, sentence length, concatenating features and label(s)
        """

        ## If Elmo we gonna use the original sentence as input.
        #if self.elmo:
         #   seqs = " ".join(self.sents[index])
          #  seq_len = len(self.sents[index])

        if not self.elmo:
            seqs = [self.word_to_ix["<s>"]] + [self.word_to_ix[word]
                    if word in self.word_to_ix else len(self.word_to_ix)
                    for word in self.sents[index]] + [self.word_to_ix["</s>"]]
                
            if self.max_length > 0:
                seq_len = len(seqs)
                if seq_len > self.max_length:
                    sent = torch.tensor(seqs[:self.max_length], dtype=torch.long)
                elif seq_len == self.max_length:
                    sent = torch.tensor(seqs, dtype=torch.long)
                else:
                    for i in range(self.max_length - seq_len):
                        seqs.append(len(self.word_to_ix) + 1) # for padding.
                        sent = torch.tensor(seqs, dtype=torch.long)
                seq_len = min(seq_len, self.max_length)
            else:
                sent = torch.tensor(seqs, dtype=torch.long)
                seq_len = len(seqs)
            add_feature = torch.tensor([(self.features.iloc[index, :])], dtype=torch.float) 
            label = torch.tensor([float(self.labels[index])], dtype=torch.float)
            if self.cuda:
                sent, seq_len, add_feature, label = sent.cuda(), seq_len, add_feature.cuda(), label.cuda()
            return sent, seq_len, add_feature, label

        elif self.elmo:
            seqs= (self.sents[index])
            seq_len = len(self.sents[index])
            
            label = torch.tensor([float(self.labels[index])], dtype=torch.float)
            add_feature = torch.tensor([(self.features.iloc[index, :])], dtype=torch.float)
            if self.cuda:
                sent, seq_len, add_features, label = seqs, seq_len, add_feature.cuda(), label.cuda()
            return sent, seq_len, add_feature, label

    def __len__(self):
        return len(self.sents)
