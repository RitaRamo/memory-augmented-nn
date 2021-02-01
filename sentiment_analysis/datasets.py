from torch.utils.data import Dataset
import faiss

class SADataset(Dataset):
        """
        A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
        """

        def __init__(self, sents_original, sents_ids, lens, labels, bert_model, max_len):
            self.sents_original = sents_original
            self.sents_ids = sents_ids
            self.lens = lens
            self.len_dataset = len(sents_ids)
            self.labels = labels
            self.model = bert_model #SentenceTransformer('paraphrase-distilroberta-base-v1')
            self.max_len = max_len
            # print("entrei no init", sents_ids)
            # print("self sents", self.sents)
            # print("self sents_tokens", self.sents_tokens)
            # print("self labels", self.labels)
            # print("self len data", self.len_dataset)

        def __getitem__(self, i):
            bert_text = self.model.encode(self.sents_original[i][:self.max_len])

            return bert_text, torch.tensor(self.sents_ids[i]).long(), torch.tensor(self.lens[i]).long(), torch.tensor(
                self.labels[i], dtype=torch.float64)

        def __len__(self):
            return self.len_dataset

class TrainRetrievalDataset(Dataset):
        """
        A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
        """

        def __init__(self, train_sents, bert_model, debug, max_len):
            if debug:
                self.train_sents=train_sents[:40] 
            else:
                self.train_sents=train_sents
            self.dataset_size = len(self.train_sents)
            self.model = bert_model #SentenceTransformer('paraphrase-distilroberta-base-v1')
            self.max_len= max_len
        
        def __getitem__(self, i):
            bert_text = self.model.encode(self.train_sents[i][:self.max_len])
            return bert_text
    
        def __len__(self):
            return self.dataset_size

class TextRetrieval():

    def __init__(self, train_dataloader, labels, model_type):
        dim_examples = 768 #size of bert embeddings
        self.datastore = faiss.IndexFlatL2(dim_examples) #datastore
        #self._add_examples(train_dataloader)
        
        if model_type== "SAR_bert":
            #if SAR_bert, more than saving all the inputs to the datastore,
            #also store a representation from the average bert embeddings of all negatives sentences, as well pos sentences
            self.labels = labels
            self.neg_bert_embedding = torch.zeros(dim_examples)
            self.pos_bert_embedding = torch.zeros(dim_examples)
            self.number_of_neg = 0.0
            self.number_of_pos = 0.0

            self._add_examples_SAR_bert(train_dataloader)
            
            self.neg_bert_embedding = self.neg_bert_embedding/self.number_of_neg
            self.pos_bert_embedding = self.pos_bert_embedding/self.number_of_pos

        else:
            self._add_examples(train_dataloader)

    def _add_examples(self, train_dataloader):
        print("\nadding input examples to datastore (retrieval)")

        for i, (text_bert) in enumerate(train_dataloader):
            
            self.datastore.add(text_bert.cpu().numpy())

            if i%5==0:
                print("batch, n of examples", i, self.datastore.ntotal)
        
        print("finish retrieval")

    def _add_examples_SAR_bert(self, train_dataloader):
        print("\nadding input examples to datastore (retrieval)")

        for i, (text_bert) in enumerate(train_dataloader):
            batch_size = text_bert.size()[0]
            
            neg_sentences = text_bert[torch.tensor(self.labels[i*batch_size:i*batch_size+batch_size])==0]
            self.number_of_neg +=neg_sentences.size()[0]
            self.neg_bert_embedding += torch.sum(neg_sentences, dim=0)
           
            pos_sentences = text_bert[torch.tensor(self.labels[i*batch_size:i*batch_size+batch_size])==1]
            self.number_of_pos += pos_sentences.size()[0]
            self.pos_bert_embedding += torch.sum(pos_sentences, dim=0)

            self.datastore.add(text_bert.cpu().numpy())

            if i%5==0:
                print("batch, n of examples", i, self.datastore.ntotal)
        
        print("finish retrieval")

    def retrieve_nearest_for_train_query(self, query_text, k=2):
        #print("self query img", query_text)
        D, I = self.datastore.search(query_text, k)     # actual search
        #print("this is I", I)
        nearest_input = I[:,1]
        #print("the actual nearest input", nearest_input)
        return nearest_input

    def retrieve_nearest_for_val_or_test_query(self, query_text, k=1):
        D, I = self.datastore.search(query_text, k)     # actual search
        nearest_input = I[:,0]
        #print("all nearest", I)
        #print("the nearest input", nearest_input)
        return nearest_input
