import torch
# from torchtext import data
# from torchtext import datasets
from sklearn.metrics import f1_score
import re
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
import torch.optim as optim
import json
from toolz.itertoolz import unique
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import faiss
import fasttext

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 32
MIN_FREQ_WORD = 5
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MAX_LEN = 500
DATA_FOLDER="dataset_splits/" #""
EMBEDDING_DIM = 300
HIDDEN_DIM = 512
OUTPUT_DIM = 1
ATTENTION_DIM = 512
N_LAYERS = 1
DROPOUT = 0.5
#device = "cpu"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_TYPE="BASELINE"
MULTI_ATTENTION = False 
DEBUG = False

class TrainRetrievalDataset(Dataset):
        """
        A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
        """

        def __init__(self, train_sents):
            if DEBUG:
                self.train_sents=train_sents[:10]
            else:
                self.train_sents=train_sents
            self.dataset_size = len(self.train_sents)
            self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')

        def __getitem__(self, i):
            bert_text = self.model.encode(self.train_sents[i][:MAX_LEN])
            return bert_text
    
        def __len__(self):
            return self.dataset_size

class TextRetrieval():

    def __init__(self, train_dataloader):
        dim_examples = 768 #size of bert embeddings
        self.datastore = faiss.IndexFlatL2(dim_examples) #datastore
        self._add_examples(train_dataloader)

    def _add_examples(self, train_dataloader):
        print("\nadding input examples to datastore (retrieval)")

        for i, (text_bert) in enumerate(train_dataloader):
            
            self.datastore.add(text_bert.cpu().numpy())

            if i%5==0:
                print("batch, n of examples", i, self.datastore.ntotal)
        
        print("finish retrieval")

    def retrieve_nearest_for_train_query(self, query_text, k=2):
        #print("self query img", query_text)
        D, I = self.datastore.search(query_text, k)     # actual search
        #print("this is I", I)
        nearest_input = I[:,1]
        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_text, k=1):
        D, I = self.datastore.search(query_text, k)     # actual search
        nearest_input = I[:,0]
        #print("all nearest", I)
        #print("the nearest input", nearest_input)
        return nearest_input

class SADataset(Dataset):
        """
        A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
        """

        def __init__(self, sents_original, sents_ids, lens, labels):
            self.sents_original = sents_original
            self.sents_ids = sents_ids
            self.lens = lens
            self.len_dataset = len(sents_ids)
            self.labels = labels
            self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')

            # print("entrei no init", sents_ids)
            # print("self sents", self.sents)
            # print("self sents_tokens", self.sents_tokens)
            # print("self labels", self.labels)
            # print("self len data", self.len_dataset)

        def __getitem__(self, i):
            bert_text = self.model.encode(self.sents_original[i][:MAX_LEN])

            return bert_text, torch.tensor(self.sents_ids[i]).long(), torch.tensor(self.lens[i]).long(), torch.tensor(
                self.labels[i], dtype=torch.float64)

        def __len__(self):
            return self.len_dataset

#################################################MODEL####################################################

class SARModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, attention_dim, n_layers, dropout,
                    pad_idx, token_to_id):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers)

        print("Model Name", MODEL_TYPE)

        if MODEL_TYPE == "SAR_-11":
            retrieved_dim= hidden_dim
        elif MODEL_TYPE == "SAR_avg":
            retrieved_dim= embedding_dim # retrieved target correspond to avg word embeddings from caption
            self.init_c = nn.Linear(retrieved_dim, hidden_dim)
        elif MODEL_TYPE == "SAR_norm":
            retrieved_dim= embedding_dim # retrieved target correspond to avg embeddings weighted by norm
        elif MODEL_TYPE == "SAR_bert":
            retrieved_dim= 768 # retrieved target correspond to bert embeddings size

        if MULTI_ATTENTION:
            print("using our multi attention")
            self.attention = self.attention_multilevel  # proposed attention network
            self.linear_retrieval = nn.Linear(hidden_dim, retrieved_dim)
            self.hiddens_att = nn.Linear(retrieved_dim, attention_dim)  # linear layer to transform hidden states
            self.cat_att = nn.Linear(retrieved_dim, attention_dim)
            self.full_multiatt = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
            self.fc = nn.Linear(retrieved_dim, output_dim)

        else: #baseline attention
            print("default attention")
            self.attention = self.attention_baseline
            self.hiddens_att = nn.Linear(hidden_dim, attention_dim)  # linear layer to transform hidden states
            self.fc = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        ############Attention###########
        self.final_hidden_att = nn.Linear(hidden_dim, attention_dim)  # linear layer to transform last hidden state
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        if DEBUG:
            print("WITHOUT FASTEXT -> DEBUG")
        else:
            print("pretraining fastext")
            #init embedding layer
            fasttext_embeddings = fasttext.load_model('../image_captioning/embeddings/wiki.en.bin')
            pretrained_embeddings = self._get_fasttext_embeddings_matrix(fasttext_embeddings, vocab_size, embedding_dim, token_to_id)
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings))

            # pretrained embedings are not trainable by default
            self.embedding.weight.requires_grad = False

    def _get_fasttext_embeddings_matrix(self,embeddings, vocab_size, embedding_dim, token_to_id):
        # reduce the matrix of pretrained:embeddings according to dataset vocab
            print("loading fasttext embeddings")

            embeddings_matrix = np.zeros(
                (vocab_size, embedding_dim))

            for word, id in token_to_id.items():
                try:
                    embeddings_matrix[id] = embeddings.get_word_vector(word)
                except:
                    print("How? Fastext has embedding for each token, hence this exception should not happen")
                    pass

            return embeddings_matrix

    def attention_multilevel(self, hiddens, final_hidden, retrieved_target):
        """
        Forward propagation.

        :param hiddens: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param final_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
       
        #the hidden features receive an affine transformation for this attention, before passing through Eq. 4,
        # to ensure that it has the same dimension of the retrieved target in order to compute Eq. 9 (combine both)
        #print("hiddens", hiddens.size())
        hiddens= self.linear_retrieval(hiddens)  #hiddens->retrieved dim
        #print("hiddens with linear retrieval", hiddens.size())

        hiddens = hiddens.permute(1, 0, 2)
        att1 = self.hiddens_att(hiddens)  # (batch_size, num_hiddens(words), attention_dim)
        att_h = self.final_hidden_att(final_hidden.permute(1, 0, 2))  # (batch_size, 1, attention_dim)
        att = self.full_att(self.tanh(att1 + att_h)).squeeze(2)  # (batch_size, num_hiddens(words), 1)
        alpha = self.softmax(att)  # (batch_size, num_hiddens(words),1)
        text_context = (hiddens * alpha.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_dim)
        #print("text context", text_context.size())
        #print("retreive target", retrieved_target.size())

        text_and_retrieved = torch.cat(([text_context.unsqueeze(1), retrieved_target.unsqueeze(1)]), dim=1)
        #print("text_and_retrieved", text_and_retrieved.size())

        att_tr= self.cat_att(text_and_retrieved) #visual with retrieved target
        #print("att_tr size", text_and_retrieved.size())

        att_hat = self.full_multiatt(self.tanh(att_tr + att_h)).squeeze(2)  # (batch_size, num_pixels)
        #print("att_hat size", text_and_retrieved.size())

        alpha_hat = self.softmax(att_hat)  # (batch_size, num_pixels)
        #print("alpha_hat size", alpha_hat.size())

        multilevel_context=(text_and_retrieved * alpha_hat.unsqueeze(2)).sum(dim=1)
        #print("multilevel contex", multilevel_context.size())
        return multilevel_context, alpha_hat

    def attention_baseline(self, hiddens, final_hidden, retrieved_target=None):
        """
        Forward propagation.
        :param hiddens: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param final_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # lstm_output = lstm_output.permute(1, 0, 2)

        # hiddens torch.Size([611, 64, 512])

        hiddens = hiddens.permute(1, 0, 2)
        # hiddens torch.Size([64, 611, 512])

        att1 = self.hiddens_att(hiddens)  # (batch_size, num_hiddens(words), attention_dim)
        # att1 torch.Size([64, 611, 512])

        # final_hidden1 torch.Size([1, 64, 512])

        att2 = self.final_hidden_att(final_hidden.permute(1, 0, 2))  # (batch_size, 1, attention_dim)
        # att2 torch.Size([64, 1, 512])

        att = self.full_att(self.tanh(att1 + att2)).squeeze(2)  # (batch_size, num_hiddens(words), 1)
        # att torch.Size([64, 611])

        alpha = self.softmax(att)  # (batch_size, num_hiddens(words),1)
        # alpha torch.Size([64, 611])
        attention_weighted_encoding = (hiddens * alpha.unsqueeze(-1)).sum(dim=1)  # (batch_size, hidden_dim)
        # attention_weighted_encoding torch.Size([64, 512])

        return attention_weighted_encoding, alpha

    def forward(self, text, text_lengths, target_neighbors_representations):
        # text = [sent len, batch size]

        # tamanho do texto, tamanho do batch
        # text torch.Size([144, 64])
        # text_lenghts torch.Size([64])
        embedded = self.embedding(text)
        # embedded torch.Size([144, 64, 100])
        # embedded = [sent len, batch size, emb dim]
        #print('embedded', embedded.size())

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to("cpu"))
        # packed_embedded torch.Size([9152, 100])
        #print("packed embedded", packed_embedded)
        #print('packed_embedded', packed_embedded.data.size())

        init_hidden, init_cell = self.init_hidden_states(target_neighbors_representations)

        packed_output, (hidden, cell) = self.lstm(packed_embedded, (init_hidden, init_cell))

        # packed_output torch.Size([9152, 512])
        #print("packed output", packed_output)
        #print('packed_output', packed_output.data.size())
        #print('hidden', hidden.size())
        #print('cell', cell.size())

        # hidden torch.Size([1, 64, 512])

        # cell torch.Size([1, 64, 512])

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output torch.Size([144, 64, 512])

        # output_lengths torch.Size([64])

        # output = [sent len, batch size, hid dim]
        #print('output', output.size())
        #print('output_lengths', output_lengths.size())

        # output over padding tokens are zero tensors

        # hidden = [num layers, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        attn_output, alpha = self.attention(output, hidden, target_neighbors_representations)

        # attn_output torch.Size([64, 512])
        # hidden = [batch size, hid dim * num directions]
        # self.fc(attn_output) torch.Size([64, 1])
        #print('attn_output', attn_output.size())
        #print('self.fc(self.dropout(attn_output)', self.fc(self.dropout(attn_output)).size())
        return self.fc(self.dropout(attn_output))

    def init_hidden_states(self, target_neighbors_representations):
        batch_size= target_neighbors_representations.size()[0]
        init_hidden = torch.zeros(batch_size, HIDDEN_DIM).to(device)

        if MODEL_TYPE== "BASELINE":
            init_cell=  target_neighbors_representations

        elif MODEL_TYPE== "SAR_-11":
            #already has dimension of the LSTM
            init_cell= target_neighbors_representations

        elif MODEL_TYPE== "SAR_avg":
            init_cell = self.init_c(target_neighbors_representations)

        else:
            raise Exception("that model does not exist")

        #print("torch init hidden", init_hidden)
        #print("torch init hidden", init_hidden.size())

        #print("torch init init_cell", init_cell)
        #print("torch init init_cell", init_cell.size())

        return init_hidden.unsqueeze(0), init_cell.unsqueeze(0)


#################################################DATA#####################################################

def main():
    ###################### PREPROCESSING##################

    with open(DATA_FOLDER+'train_sents.json', 'r') as j:
        train_sents = json.load(j)
    with open(DATA_FOLDER+'train_labels.json', 'r') as j:
        train_labels = json.load(j)

    with open(DATA_FOLDER+'test_sents.json', 'r') as j:
        test_sents = json.load(j)
    with open(DATA_FOLDER+'test_labels.json', 'r') as j:
        test_labels = json.load(j)

    #print("train sets in beging", len(train_sents))
    train_sents, val_sents, train_labels, val_labels = train_test_split(train_sents,train_labels, test_size=0.1, random_state=42)
    #print("train sets after", len(train_sents))

    train_words = " ".join(train_sents).split()
    words_counter = Counter(train_words)
    words = [w for w in words_counter.keys() if words_counter[w] > MIN_FREQ_WORD]
    # print("our words", words)
    # print("len words", len(words))

    vocab = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN] + words
    # print("vocab", vocab)
    #print("vocab", len(vocab))

    token_to_id = OrderedDict([(value, index)
                               for index, value in enumerate(vocab)])
    id_to_token = OrderedDict([(index, value)
                               for index, value in enumerate(vocab)])

    #print("token_to_id", token_to_id)

    model = SARModel(len(vocab),
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     ATTENTION_DIM,
                     N_LAYERS,
                     DROPOUT,
                     token_to_id[PAD_TOKEN],
                     token_to_id)


    ##############################DATA###############################

    def convert_sent_tokens_to_ids(sent_of_tokens, max_len, token_to_id):
        len_sents = []
        # +2 to account for start and end token
        input_sents = np.zeros((len(sent_of_tokens), max_len + 2), dtype=np.int32) + token_to_id[PAD_TOKEN]

        for i in range(len(sent_of_tokens)):
            tokens_to_integer = [token_to_id.get(token, token_to_id[UNK_TOKEN]) for token in sent_of_tokens[i]]

            sent = tokens_to_integer[:max_len]

            sent_with_spetial_tokens = [token_to_id[START_TOKEN]] + sent + [token_to_id[END_TOKEN]]

            input_sents[i, :len(sent_with_spetial_tokens)] = sent_with_spetial_tokens

            len_sents.append(len(sent_with_spetial_tokens))

        return input_sents, len_sents

    sents_with_tokens = [text.split() for text in train_sents]
    train_sents_ids, train_lens = convert_sent_tokens_to_ids(sents_with_tokens, MAX_LEN, token_to_id)

    #print("train_sents_ids", len(train_sents_ids))

    val_sents_with_tokens = [text.split() for text in val_sents]
    val_sents_ids, val_lens = convert_sent_tokens_to_ids(val_sents_with_tokens, MAX_LEN, token_to_id)

    test_sents_with_tokens = [text.split() for text in test_sents]
    test_sents_ids, test_lens = convert_sent_tokens_to_ids(test_sents_with_tokens, MAX_LEN, token_to_id)


    #print("INPUT sents", train_sents_ids)


    #print("train sents", train_sents[:10])
    #model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    #bert_text = torch.tensor(model.encode(train_sents[:10])).to(device)

    #RETRIEVAL: add to datastore the inputs text
    train_retrieval_iterator = torch.utils.data.DataLoader(
        TrainRetrievalDataset(train_sents),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    text_retrieval=TextRetrieval(train_retrieval_iterator)
    target_lookup = torch.tensor(train_labels)

    print("loading the two representations for the binary targets")
    if MODEL_TYPE == "BASELINE":
        # the baseline does not have retrieved target
        #it is just for code coherence in respect to SAR Model that needs it for the lstm init states
        target_representations= torch.zeros(2, HIDDEN_DIM).to(device)

    elif MODEL_TYPE== "SAR_-11":
        target_representations= torch.ones(2, HIDDEN_DIM).to(device)
        target_representations[0,:] = -1.0 
        #target representations is a vector of either -1 or 1s
        #torch.tensor([-1*torch.ones(BATCH_SIZE, HIDDEN_DIM), torch.ones(BATCH_SIZE, HIDDEN_DIM)]).to(device)
   
    elif MODEL_TYPE == "SAR_avg":
        #temos de ter todas as frases negativas
        #passar essas frases com os ids pela layer embedding
        #do mean()
        print("entrei no SAR", )
        #train_sents_ids = torch.tensor(train_sents_ids)

        #train_labels=torch.tensor(train_labels)
        train_neg_sents_ids=torch.tensor(train_sents_ids)[torch.tensor(train_labels)==0]
        train_pos_sents_ids=torch.tensor(train_sents_ids)[torch.tensor(train_labels)==1]
       
        negs_embeddings=model.embedding(train_neg_sents_ids[:10].long())
        pos_embeddings=model.embedding(train_pos_sents_ids[:10].long())

        print("neg embeddings size", negs_embeddings.size())

        avg_negs_embedding=negs_embeddings.mean(1).mean(0)
        avg_pos_embedding=pos_embeddings.mean(0)
        print("torch size", avg_negs_embedding.size())

        target_representations= torch.cat((avg_negs_embedding.unsqueeze(0), avg_negs_embedding.unsqueeze(0)), dim=0)
        print("target repres", target_representations.size())
        #target_representations= torch.ones(2, EMBEDDING_DIM).to(device)
        #target_representations[0,:]= avg_negs_embedding
        #target_representations[1,:]= avg_negs_embedding

        #print(stop)

    else:
        raise Exception("Unknown model")
    
    # elif MODEL_TYPE== "SAR_norm":
    #     train_sents_ids = torch.tensor(train_sents_ids)
    #     train_labels=torch.tensor(train_labels)
    #     train_neg_sents_ids=train_sents_ids[train_labels==0]
    #     train_pos_sents_ids=train_sents_ids[train_labels==1]

    #     negs_embedding=model.embedding(train_neg_sents_ids.long())
    #     pos_embeddings=model.embedding(train_pos_sents_ids.long())



        #SIMILAR IDEA

    #else: #model bert
        #train_neg_sents_original= # 
        #train_pos_sents_original= #

  

    # print("new val_sents sents", val_sents[0])
    # print("val_sents ids", val_sents_ids[0,:])

    # print(stop)

    # train_sents_ids, val_sents_ids, train_lens, val_lens, train_labels, val_labels = train_test_split(train_sents_ids,
    #                                                                                                   train_lens,
    #                                                                                                   train_labels,
    #                                                                                                   test_size=0.1,
    #                                                                                                   random_state=42)

    #DEBUG
    # train_iterator = torch.utils.data.DataLoader(
    #     SADataset(train_sents, train_sents_ids, train_lens, train_labels),
    #     batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    # )

    # print("train sents", train_sents[])

    # for i, (sents_bert, sents, lens, labels) in enumerate(train_iterator):
    #     print("batch i", i)
    #     print("sent_bert", sents_bert)
    #     print("sents", sents)
    #     print("len", lens)
    #     print("labels", labels)
    #     # 4 primeiras frases 
    #     #pedir ao retrieval para cada uma destas frases o seu vizinho
    #     # [0,1,2,3]
    #     retrieved_neighbors_index = text_retrieval.retrieve_nearest_for_train_query(sents_bert.numpy())
    #     print("nearest_input", retrieved_neighbors_index)
    #     target_neighbors=target_lookup[retrieved_neighbors_index]
    #     print("target_neighbors", target_neighbors)
    #     print(stop)
    
    #####
    train_iterator = torch.utils.data.DataLoader(
        SADataset(train_sents, train_sents_ids, train_lens, train_labels),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_iterator = torch.utils.data.DataLoader(
        SADataset(val_sents, val_sents_ids, val_lens, val_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_iterator = torch.utils.data.DataLoader(
        SADataset(test_sents, test_sents_ids, test_lens, test_labels),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )


    ############################################TRAIN#################################################

    def adjust_learning_rate(optimizer, shrink_factor):
        print("\nDECAYING learning rate.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
        print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def binary_accuracy(preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def f1score(preds, y):
        predictions = torch.round(torch.sigmoid(preds))
        return f1_score(y.cpu().detach().numpy(), predictions.cpu().detach().numpy())

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0

        model.train()

        for batch, (text_bert, text, text_lengths, label) in enumerate(iterator):
            optimizer.zero_grad()

            retrieved_neighbors_index = text_retrieval.retrieve_nearest_for_train_query(text_bert.numpy()) 
            target_neighbors=target_lookup[retrieved_neighbors_index]
            #print("target_neighbors", target_neighbors)
            #print("target_neighbors", target_representations)

            target_neighbors_representations = target_representations[target_neighbors]
            #print("target neighr represntations",target_neighbors_representations )
            #print("target neighr represntations",target_neighbors_representations.size())

            # text, text_lengths = batch.text
            text_lengths, sort_ind = text_lengths.sort(dim=0, descending=True)
            text_lengths = text_lengths.to(device)
            text = text[sort_ind].to(device)
            label = label[sort_ind].to(device)       
            target_neighbors_representations = target_neighbors_representations[sort_ind].to(device) 
            text= text.permute(1, 0)
          
            #TODO: init_hidden_states():
            predictions = model(text, text_lengths, target_neighbors_representations).squeeze(1)

            #print("labels", label)
            #print("labels long", label.long())

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            f1 = f1score(predictions, label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()

            if batch %20==0:
                print(f'\tTrain Loss: {(epoch_loss/ (batch+1)):.4f} | Train Acc: {(epoch_acc/ (batch+1)) * 100:.4f}% | Train f1-score {(epoch_f1/ (batch+1)):.4f}')

            #TODO: REMOVER
            #break

        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        model.eval()

        with torch.no_grad():
            for batch, (text_bert, text, text_lengths, label) in enumerate(iterator):

                retrieved_neighbors_index = text_retrieval.retrieve_nearest_for_train_query(text_bert.numpy()) 
                target_neighbors=target_lookup[retrieved_neighbors_index]
                #print("target_neighbors", target_neighbors)
                #print("target_neighbors", target_representations)

                target_neighbors_representations = target_representations[target_neighbors]
                #print("target neighr represntations",target_neighbors_representations )
                #print("target neighr represntations",target_neighbors_representations.size())

                # text, text_lengths = batch.text
                text_lengths, sort_ind = text_lengths.sort(dim=0, descending=True)
                text_lengths = text_lengths.to(device)
                text = text[sort_ind].to(device)
                label = label[sort_ind].to(device)       
                target_neighbors_representations = target_neighbors_representations[sort_ind].to(device) 
                text= text.permute(1, 0)
          
                predictions = model(text, text_lengths, target_neighbors_representations).squeeze(1)

                loss = criterion(predictions, label)

                f1 = f1score(predictions, label)

                acc = binary_accuracy(predictions, label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_f1 += f1.item()

                if batch %20==0:
                    print(f'\VAL (or test) Loss: {(epoch_loss/ (batch+1)):.4f} | VAL Acc: {(epoch_acc/ (batch+1)) * 100:.4f}% | VAL f1-score {(epoch_f1/ (batch+1)):.4f}')

                #TODO: REMOVER
                #break

        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHS = 20
    best_valid_acc = 0
    counter_without_improvement = 0

    for epoch in range(N_EPOCHS):

        if counter_without_improvement == 12:
            break

        if counter_without_improvement > 0 and counter_without_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        start_time = time.time()

        train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, valid_f1 = evaluate(model, val_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc:
            counter_without_improvement = 0
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'tut2-model.pt')
        else:
            counter_without_improvement += 1

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.4f}% | Train f1-score {train_f1:.4f}')
        print(f'\t Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc * 100:.4f}% | Val. f1-score {valid_f1:.4f}')
        #break

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc, test_f1 = evaluate(model, test_iterator, criterion)

    print("Model Name", MODEL_TYPE)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.4f}% | Test f1-score {test_f1:.4f} ')
    print("Test entire value", test_acc, test_f1)
    #: 01 | Epoch Time: 0m 34s#
    #	Train Loss: 0.666 | Train Acc: 59.25% | Train f1-score 0.528
    #	 Val. Loss: 0.783 |  Val. Acc: 53.55% | Val. f1-score 0.167
    # Epoch: 02 | Epoch Time: 0m 34s
    #	Train Loss: 0.574 | Train Acc: 69.99% | Train f1-score 0.660
    #	 Val. Loss: 0.467 |  Val. Acc: 79.59% | Val. f1-score 0.794


if __name__ == '__main__':
    main()