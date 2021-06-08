import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
START_TAG = "<START>"
STOP_TAG = "<STOP>"
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class bilstm_crf(nn.Module):
    def __init__(self, args, tag_to_ix):
        super(bilstm_crf, self).__init__()
        self.embedding_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.args = args
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.dropout = args.dropout
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=self.dropout, bidirectional = True)

        self.hidden2tag = nn.Linear(self.hidden_dim*2,  self.tagset_size)

        # Matrix of transition parameters.
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden(args)

    def init_hidden(self, args):
        if (args.use_gpu):
            return (torch.randn(2, args.batch_size, self.hidden_dim).cuda(),
                torch.randn(2, args.batch_size, self.hidden_dim).cuda())
        return (torch.randn(2, args.batch_size, self.hidden_dim),
                torch.randn(2, args.batch_size, self.hidden_dim))

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas

        # interate documents.
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
             
                emit_score = feat[next_tag].view(1,-1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1,-1)
                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            foward_var = torch.cat(alphas_t).view(1,-1)
        terminal_var = foward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        
        return alpha

    def _get_lstm_features(self, document):
        self.hidden = self.init_hidden(self.args)
        batch_size, seq_len, embed_dim = document.shape
        embed = torch.FloatTensor(document).view(seq_len, batch_size, embed_dim)

        if self.args.use_gpu:
            embed = embed.cuda()

        lstm_out, self.hidden = self.lstm(embed, self.hidden)
        lstm_out = lstm_out.view(seq_len, self.hidden_dim*2)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        ## Give the score of a tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i+1],tags[i]] + feat[tags[i+1]]
        score += self.transitions[self.tag_to_ix[STOP_TAG],tags[-1]]
        return score

    def _vitervi_decode(self, feats):

        backpointers = []

        init_vvar = torch.full((1, self.tagset_size), -100000000)
        init_vvar[0][self.tag_to_ix[START_TAG]] = 0
        print("initial_VVAR: ", init_vvar)
        forward_var = init_vvar
        for feat in feats:
        #    print(feat)
            bptr_t = []
            viterbivar_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                #print("next tag", next_tag_var)
                best_tag_id = argmax(next_tag_var)
               # print(best_tag_id)
               
                bptr_t.append(best_tag_id)
                #print(bptr_t)
                viterbivar_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivar_t)+feat).view(1,-1)
            backpointers.append(bptr_t)
        #print(backpointers)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
       # print(terminal_var)
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        #decode best path
        best_path = [best_tag_id]
        for bptr_t in reversed(backpointers):
            best_tag_id = bptr_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, article, tags):
        feats = self._get_lstm_features(article)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
       # print(gold_score, forward_score)
        return forward_score - gold_score

    def forward(self, article):
        lstm_feats = self._get_lstm_features(article)

        score, tag_seq = self._vitervi_decode(lstm_feats)
        return score, tag_seq









