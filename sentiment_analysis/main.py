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

seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#################################################DATA#####################################################

MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 32
MIN_FREQ_WORD = 5
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
MAX_LEN = 500


def main():
    # torch.multiprocessing.freeze_support()
    # 1- vocabulario
    #
    with open('train_sents.json', 'r') as j:
        train_sents = json.load(j)
    with open('train_labels.json', 'r') as j:
        train_labels = json.load(j)

    # print("train sents yeah", train_sents)

    train_words = " ".join(train_sents).split()
    words_counter = Counter(train_words)
    words = [w for w in words_counter.keys() if words_counter[w] > MIN_FREQ_WORD]
    # print("our words", words)
    # print("len words", len(words))

    vocab = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN] + words
    # print("vocab", vocab)
    print("vocab", len(vocab))

    token_to_id = OrderedDict([(value, index)
                               for index, value in enumerate(vocab)])
    id_to_token = OrderedDict([(index, value)
                               for index, value in enumerate(vocab)])

    sents_with_tokens = [text.split() for text in train_sents]

    print("token_to_id", token_to_id)

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

    train_sents_ids, train_lens = convert_sent_tokens_to_ids(sents_with_tokens, MAX_LEN, token_to_id)
    print("INPUT sents", train_sents_ids)

    # print("len_sents", train_lens)

    class SADataset(Dataset):
        """
        A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
        """

        def __init__(self, sents_ids, lens, labels):
            self.sents_ids = sents_ids
            self.lens = lens
            self.len_dataset = len(sents_ids)
            self.labels = labels

            print("entrei no init", sents_ids)
            # print("self sents", self.sents)
            # print("self sents_tokens", self.sents_tokens)
            # print("self labels", self.labels)
            # print("self len data", self.len_dataset)

        def __getitem__(self, i):
            return torch.tensor(self.sents_ids[i]).long(), torch.tensor(self.lens[i]).long(), torch.tensor(
                self.labels[i]).long()

        def __len__(self):
            return self.len_dataset

    train_sents_ids, train_lens, train_labels, val_sents_ids, val_lens, val_labels = train_test_split(train_sents_ids,
                                                                                                      train_lens,
                                                                                                      train_labels,
                                                                                                      test_size=0.1,
                                                                                                      random_state=41)

    # TODO: SUFFLE TRUE!!!!
    train_iterator = torch.utils.data.DataLoader(
        SADataset(train_sents_ids, train_lens, train_labels),
        batch_size=4, shuffle=False, num_workers=0
    )

    val_iterator = torch.utils.data.DataLoader(
        SADataset(val_sents_ids, val_lens, val_labels),
        batch_size=4, shuffle=False, num_workers=0
    )

    for i, (sents, lens, labels) in enumerate(val_iterator):
        print("batch i", i)
        print("sents", sents)
        print("len", lens)
        print("labels", labels)
        print(stop)

    # class TrainRetrievalDataset(Dataset):
    #     """
    #     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    #     """

    #     def __init__(self, sents, labels):
    #       self.sents=sents
    #       self.labels=labels
    #       print("entrei no init")

    #     def __getitem__(self, i):
    #       print("entrei aqui no get item")
    #       return torch.tensor([1,2,3])

    #     def __len__(self):
    #       #print("this is the actual len on __len", self.dataset_size)
    #     #   sent_text= sents[i]
    #     #   tokens_to_integer = [token_to_id.get(
    #     #         token, token_to_id[UNK_TOKEN]) for token in sent_text[i]]

    #     #    input_sent = np.zeros(
    #     #     (1, max_len)) + token_to_id[PAD_TOKEN]

    #     #    input_sent = 

    #       return 

    # TODO: SEM ESQUECER DO SLIP!!

    # print(stop)
    # def tokenizer(doc):
    #     return [i.lower() for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", doc) if
    #             i != '' and i != ' ' and i != '\n']

    # TEXT = data.Field(tokenize=tokenizer, include_lengths=True)

    # LABEL = data.LabelField(dtype=torch.float)

    # train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    # train_data, valid_data = train_data.split(random_state=random.seed(seed))

    # TEXT.build_vocab(train_data,
    #                  max_size=MAX_VOCAB_SIZE,
    #                  vectors="glove.6B.300d",
    #                  unk_init=torch.Tensor.normal_)

    # LABEL.build_vocab(train_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    #     (train_data, valid_data, test_data),
    #     batch_size=BATCH_SIZE,
    #     sort_within_batch=True,
    #     device=device)

    # index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), self.dim)

    #################################################MODEL####################################################

    class SARModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, attention_dim, n_layers, dropout,
                     pad_idx):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

            self.rnn = nn.LSTM(embedding_dim,
                               hidden_dim,
                               num_layers=n_layers)

            self.fc = nn.Linear(attention_dim, output_dim)

            self.dropout = nn.Dropout(dropout)
            ############Attention###########
            self.hiddens_att = nn.Linear(hidden_dim, attention_dim)  # linear layer to transform hidden states
            self.final_hidden_att = nn.Linear(hidden_dim, attention_dim)  # linear layer to transform last hidden state
            self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
            self.tanh = nn.Tanh()
            self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

        #    def attention(self, lstm_output, final_state):
        #        #lstm_output1 torch.Size([144, 64, 512])
        #
        #       lstm_output = lstm_output.permute(1, 0, 2)
        #        #lstm_output2 torch.Size([64, 144, 512])
        #
        #        merged_state = torch.cat([s for s in final_state], 1)
        #        #merged_state1 torch.Size([64, 512])
        #
        #        merged_state = merged_state.squeeze(0).unsqueeze(2)
        #        #merged_state2 torch.Size([64, 512, 1])
        #
        #        weights = torch.bmm(lstm_output, merged_state)
        #        #weights1 torch.Size([64, 144, 1])
        #        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        #        #weights2 torch.Size([64, 144, 1])
        #        #torch.bmm(torch.transpose(lstm_output, 1, 2), weights) torch.Size([64, 512, 1])
        #        #torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2) torch.Size([64, 512])
        #        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

        def attention(self, hiddens, final_hidden):
            """
            Forward propagation.
            :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
            :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
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

        def forward(self, text, text_lengths):
            # text = [sent len, batch size]

            # text torch.Size([144, 64])

            # text_lenghts torch.Size([64])

            embedded = self.dropout(self.embedding(text))
            # embedded torch.Size([144, 64, 100])
            # embedded = [sent len, batch size, emb dim]
            print('embedded', embedded.shape())

            # pack sequence
            device = "cpu"
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to(device))
            # packed_embedded torch.Size([9152, 100])
            print('packed_embedded', packed_embedded.shape())

            packed_output, (hidden, cell) = self.rnn(packed_embedded)
            # packed_output torch.Size([9152, 512])
            print('packed_output', packed_output.shape())
            print('hidden', hidden.shape())
            print('cell', cell.shape())

            # hidden torch.Size([1, 64, 512])

            # cell torch.Size([1, 64, 512])

            # unpack sequence
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

            # output torch.Size([144, 64, 512])

            # output_lengths torch.Size([64])

            # output = [sent len, batch size, hid dim]
            print('output', output.shape())
            print('output_lengths', output_lengths.shape())

            # output over padding tokens are zero tensors

            # hidden = [num layers, batch size, hid dim]

            # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
            # and apply dropout

            attn_output, alpha = self.attention(output, hidden)
            # attn_output torch.Size([64, 512])
            # hidden = [batch size, hid dim * num directions]
            # self.fc(attn_output) torch.Size([64, 1])
            print('attn_output', attn_output.shape())
            print('self.fc(self.dropout(attn_output)', self.fc(self.dropout(attn_output).shape()))
            return self.fc(self.dropout(attn_output))

    ############################################TRAIN#################################################

    def adjust_learning_rate(optimizer, shrink_factor):
        print("\nDECAYING learning rate.")
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
        print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

    INPUT_DIM = len(vocab)
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 512
    OUTPUT_DIM = 1
    ATTENTION_DIM = 512
    N_LAYERS = 1
    DROPOUT = 0.5
    PAD_IDX = token_to_id[PAD_TOKEN]

    model = SARModel(INPUT_DIM,
                     EMBEDDING_DIM,
                     HIDDEN_DIM,
                     OUTPUT_DIM,
                     ATTENTION_DIM,
                     N_LAYERS,
                     DROPOUT,
                     PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    # pretrained_embeddings = TEXT.vocab.vectors

    # print(pretrained_embeddings.shape)

    # model.embedding.weight.data.copy_(pretrained_embeddings)

    # UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    # model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    # model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    # print(model.embedding.weight.data)

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

        for batch, (text, text_lengths, label) in enumerate(iterator):
            optimizer.zero_grad()

            # text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, label)

            acc = binary_accuracy(predictions, label)

            f1 = f1score(predictions, label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0
        epoch_f1 = 0
        model.eval()

        with torch.no_grad():
            for batch, (text, text_lengths, label) in enumerate(iterator):
                # text, text_lengths = batch.text

                predictions = model(text, text_lengths).squeeze(1)

                loss = criterion(predictions, label)

                f1 = f1score(predictions, label)

                acc = binary_accuracy(predictions, label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_f1 += f1.item()

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

        if counter_without_improvement == 20:
            break

        if counter_without_improvement > 0 and counter_without_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        start_time = time.time()

        train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_acc > best_valid_acc:
            counter_without_improvement = 0
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), 'tut2-model.pt')
        else:
            counter_without_improvement += 1

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train f1-score {train_f1:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}% | Val. f1-score {valid_f1:.3f}')

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc, test_f1 = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}% | Test f1-score {test_f1:.3f} ')

    #: 01 | Epoch Time: 0m 34s#
    #	Train Loss: 0.666 | Train Acc: 59.25% | Train f1-score 0.528
    #	 Val. Loss: 0.783 |  Val. Acc: 53.55% | Val. f1-score 0.167
    # Epoch: 02 | Epoch Time: 0m 34s
    #	Train Loss: 0.574 | Train Acc: 69.99% | Train f1-score 0.660
    #	 Val. Loss: 0.467 |  Val. Acc: 79.59% | Val. f1-score 0.794


if __name__ == '__main__':
    main()