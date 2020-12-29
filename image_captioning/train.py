# Test1

import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from sentence_transformers import SentenceTransformer
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Data parameters

# folder with data files saved by create_input_files.py
data_folder = 'dataset_splits'
# base name shared by data files
data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim = 300  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5


# sets device for model and PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
# set to true only if inputs to model are fixed size; otherwise lot of computational overhead
cudnn.benchmark = True

# Training parameters
start_epoch = 0
# number of epochs to train for (if early stopping is not triggered)
epochs = 100
# keeps track of number of epochs since there's been an improvement in validation BLEU
epochs_since_improvement = 0
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none

MULTILEVEL_ATTENTION = False
MODEL_TYPE = "SAR_norm"
#BASELINE
#SAR_avg
#SAR_norm
#SAR_bert 

print("batch size and epochs", batch_size, epochs)

def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    if MODEL_TYPE=="SAR_bert": #with bert you need the original text (before having ids)

        with open(os.path.join(data_folder, 'TRAIN'+'_TEXT_FIRST_CAPTIONS_' + data_name + '.json'), 'r') as j:
            train_text_caps = list(json.load(j))
            #print("retrieval lookup table", len(train_text_caps))
            model = SentenceTransformer('paraphrase-distilroberta-base-v1')
            retrieval_lookup_table= torch.tensor(model.encode(train_text_caps)).to(device)
            #print("retrieval lookup table with bert", retrieval_lookup_table)
    else:
        with open(os.path.join(data_folder, 'TRAIN'+'_CAPTIONS_' + data_name + '.json'), 'r') as j:
            retrieval_lookup_table = torch.tensor(json.load(j)).to(device)
            
            

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(model_type=MODEL_TYPE,
                                       multi_attention = MULTILEVEL_ATTENTION,
                                       attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       token_to_id = word_map,
                                       lookup_table=retrieval_lookup_table,
                                       dropout=dropout,
                                       )
                                       
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)  

    # Custom dataloaders

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers)#, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers)#, pin_memory=True)

    # TrainRetrievalDataset

    train_retrieval_loader = torch.utils.data.DataLoader(
        TrainRetrievalDataset(data_folder, data_name),
        batch_size=batch_size, shuffle=True, num_workers=workers)#, pin_memory=True)

    image_retrieval = ImageRetrieval(decoder.encoder_dim, encoder, train_retrieval_loader, device)

    # print("\nagora vou entrar no my dataset")

    # diff_train_loader = torch.utils.data.DataLoader(
    #     NearestCaptionAvgDataset(data_folder, data_name, 'TRAIN'),
    #     batch_size=batch_size, shuffle=True, num_workers=workers)#, pin_memory=True)
    
    # with open(os.path.join(data_folder,   'TRAIN'+'_CAPTIONS_' + data_name + '.json'), 'r') as j:
    #         all_captions = json.load(j)

    # print("all caps", all_captions[:10])
    # all_captions= torch.tensor(all_captions)
    # print("converted all caption to tensor", all_captions.size())
    # print("converted all caption to tensor", all_captions[:10])


    # for i, (imgs, caps, caplens, ids_dataloader) in enumerate(diff_train_loader):
    #     print("\n COMECA AQUI i of batch",i)
    #     print("IMAGES size", imgs.size())
    #     print("IMAGES WITH NUMP size", np.shape(imgs.numpy()))
    #     encoder_output = encoder(imgs)
    #     encoder_output = encoder_output.view(encoder_output.size()[0], -1, encoder_output.size()[-1])
    #     print("this was the encod out", encoder_output.size())
    #     input_imgs = encoder_output.mean(dim=1)

    #     nearest=retrieval.retrieve_nearest_for_train_query(input_imgs.numpy())
  
    #     print("my final nearest", nearest)
    
    #     print("ids do dataloader", ids_dataloader)
    #     print("this were caps of input", caps)

    #     print("* 5,",nearest*5 ) #*5 sincete id that comes is the n of image, and to get the given caption we need to multiply by 5 
    #     print("final target", all_captions[nearest*5])
    #     print(stop)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 12:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              retrieval=image_retrieval,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                retrieval=image_retrieval,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" %
                  (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name + "_"+MODEL_TYPE+str(MULTILEVEL_ATTENTION), epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        #break

def train(train_loader, encoder, decoder, retrieval, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        imgs = imgs.view(imgs.size()[0], -1, imgs.size()[-1])
        #print("this was the imgs out", imgs.size())
        input_imgs = imgs.mean(dim=1)
        nearest_imgs = retrieval.retrieve_nearest_for_train_query(input_imgs.cpu().numpy())

        scores, caps_sorted, decode_lengths, sort_ind = decoder(
            imgs, nearest_imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(
            scores, decode_lengths, batch_first=True)
        targets = pack_padded_sequence(
            targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores.data, targets.data)

        # Add doubly stochastic attention regularization
        #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores.data, targets.data, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
        #break

def validate(val_loader, encoder, decoder, retrieval, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            imgs = encoder(imgs)
            imgs = imgs.view(imgs.size()[0], -1, imgs.size()[-1])
            input_imgs = imgs.mean(dim=1)
            nearest_imgs=retrieval.retrieve_nearest_for_val_or_test_query(input_imgs.cpu().numpy())

            scores, caps_sorted, decode_lengths, sort_ind = decoder(
                imgs, nearest_imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(
                scores, decode_lengths, batch_first=True)
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores.data, targets.data)

            # Add doubly stochastic attention regularization
            #loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores.data, targets.data, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # because images were sorted in the decoder
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

            #print(stop)
            #break

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
