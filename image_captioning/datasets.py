import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import faiss                   # make faiss available

#class Faiss
#class CaptionDatasetAvg()
#class CaptionDatasetAvg()

# d = self.encoder.encoder_dim
# index = faiss.IndexFlatL2(d)
# images_ids = []

# train_dataset = get_dataset(PATH_DATASETS_RSICD + "train_coco_format.json")
# #train_dataset = get_dataset(PATH_DATASETS_RSICD + "val_coco_format.json")

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
#                          std=[0.229, 0.224, 0.225])
# ])

# for values in train_dataset["images"]:

#     img_name = values["file_name"]
#     image_id = values["id"]

#     image_name = PATH_RSICD + \
#         "raw_dataset/RSICD_images/" + img_name
#     image = cv2.imread(image_name)
#     image = transform(image)
#     image = image.unsqueeze(0)

#     images_ids.append(image_id)

#     encoder_output = self.encoder(image)
#     encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])
#     mean_encoder_output = encoder_output.mean(dim=1)
#     index.add(mean_encoder_output.numpy())

# dict_imageid_refs = defaultdict(list)
# all_captions = []
# for ref in train_dataset["annotations"]:
#     image_id = ref["image_id"]
#     caption = ref["caption"]
#     all_captions.append(caption)
#     dict_imageid_refs[image_id].append(caption)

# counter_refs = Counter(all_captions)


class ImageRetrieval():

    def __init__(self, dim_examples, data_folder, data_name, encoder):
        self.index = faiss.IndexFlatL2(dim_examples) #datastore
        self.lookup_targets = {}
        #data
        with open(os.path.join(data_folder, 'TRAIN' + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            target_examples = json.load(j) 

        with open(os.path.join(data_folder, 'TRAIN' + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            input_examples = json.load(j) 

        self.data_folder=data_folder
        self._add_examples(input_examples, target_examples)

        self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                        std=[0.229, 0.224, 0.225])
        ])

        self.encoder= encoder

    def _add_examples(self, input_examples, target_examples):
        input_id = 0
        for i in input_examples:
            img_path = input_examples[i]
            img_captions = target_examples[i+5] #5 captions

            #convert to high dimensional vector
            img = Image.open(self.data_folder+"/"+img_path) #complete_path of image
            img = self.transform(img)

            encoder_output = self.encoder(image)
            encoder_output = encoder_output.view(1, -1, encoder_output.size()[-1])
            input_image = encoder_output.mean(dim=1)

            self.lookup_targets[input_id] = img_caption

            #add to the datastore
            index.add(input_image.numpy())
            input_id+=1

    def retrieve_nearest(self, query_img, k=1):
        D, I = index.search(query_img, k)     # actual search
        nearest_target = self.lookup_targets[I[0,0]]
        return nearest_target


        #return I #nearest image

        #self.lookup_targets[I] # target




class NearestCaptionAvgDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder=data_folder
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Captions per image
        self.cpi = 5

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load image_id (completely into memory)
        with open(os.path.join(data_folder, self.split + '_IMGIDS_' + data_name + '.json'), 'r') as j:
            self.imgids = json.load(j)

        with open(os.path.join(data_folder, self.split + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            self.imgpaths = json.load(j)

        # Total number of datapoints
        self.dataset_size = len(self.captions)
        
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                     std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = Image.open(self.data_folder+"/"+self.imgpaths[i // self.cpi])


        img = self.transform(img)

        #nearest_image = retrival(image)
        #nearest_caption 
        #nearest caption -> média dos embeddings
        #nearest = média dos embeddings
        
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, nearest, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            if self.split == "TEST":
                img_id = self.imgids[i]
                return img, nearest, caption, caplen, all_captions, img_id
            else:
                return img, nearest, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.data_folder=data_folder
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        # self.h = h5py.File(os.path.join(
        #     data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        # self.imgs = self.h['images']

        # Captions per image
        self.cpi = 5

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load image_id (completely into memory)
        with open(os.path.join(data_folder, self.split + '_IMGIDS_' + data_name + '.json'), 'r') as j:
            self.imgids = json.load(j)

        with open(os.path.join(data_folder, self.split + '_IMGPATHS_' + data_name + '.json'), 'r') as j:
            self.imgpaths = json.load(j)


        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = Image.open(self.data_folder+"/"+self.imgpaths[i // self.cpi])
  
        self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # mean=IMAGENET_IMAGES_MEAN, std=IMAGENET_IMAGES_STD
                                     std=[0.229, 0.224, 0.225])
        ])

        img = self.transform(img)
        
        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            if self.split == "TEST":
                img_id = self.imgids[i]
                return img, caption, caplen, all_captions, img_id
            else:
                return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
