import imp
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os

# Construct the dataset.
class TwitterDataset(Dataset):
    def __init__(
        self,
        tweets: np.array,
        labels: np.array,
        sentiment_targets: np.array,
        image_ids: np.array,
        image_captions,
        imageid2triple,
        tokenizer,
        max_len: int,
        dataset,
        triple_number,
    ):
        """
        Downstream code expects reviews and targets to be NumPy arrays.
        """
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.sentiment_targets = sentiment_targets
        self.image_captions = image_captions
        self.imageid2triple = imageid2triple
        self.max_len = max_len
        self.image_ids = image_ids
        self.sentimentlabelDict = {
            0: "negative",
            1: "neutral",
            2: "positive"
        }
        self.dataset = dataset
        self.triple_number = triple_number
        self.image_path = "data/IJCAI2019_data/twitter2015_images/".replace("twitter2015", self.dataset)
        self.sub_image_path = "cache/sub_twitter2015_images".replace("twitter2015", self.dataset)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize,
                                  ])

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        sentiment_target = self.sentiment_targets[item]
        tweet = str(self.tweets[item]).replace("RT @","").replace("$T$",'<target>'+sentiment_target+'</target>').replace("#","")
        try:
            caption = self.image_captions[self.image_ids[item]]
        except KeyError:  # A couple of the images have no content.
            caption = ""

        # seralize the triples
        try:
            triples = self.imageid2triple[self.image_path +self.image_ids[item]]
            triple_sentence = ""
            special_triple_sentence = ""
            image_path = list()
            image_triple = list()
            text_triple = list()
            for i, triple in enumerate(triples):
                if triple[1] == 'image of':
                    image_triple.append(triple)
                else:
                    text_triple.append(triple)

            image_triple_num = self.triple_number
            text_triple_num = self.triple_number
            image_triple = image_triple[:image_triple_num]
            text_triple = text_triple[:image_triple_num]
            if len(image_triple) < image_triple_num:
                for i in range(image_triple_num - len(image_triple)):
                    image_triple.append(image_triple[i])
            if len(text_triple) < text_triple_num:
                for i in range(text_triple_num - len(text_triple)):
                    text_triple.append(text_triple[i])

            entAndRel2id = dict()
            id = 1
            for triple in image_triple:
                h,r,t = triple
                if self.dataset == "twitter2015": 
                    image_path.append(os.path.join(self.sub_image_path, h +'.png'))
                else:
                    image_path.append(os.path.join(self.sub_image_path, h))
                sub_sentence = '<image>' + ',' + r + ',' + t + '<ts>'
                triple_sentence = triple_sentence + sub_sentence

                if r not in entAndRel2id:
                    entAndRel2id[r] = id 
                    id = id + 1
                if t not in entAndRel2id:
                    entAndRel2id[t] = id
                    id = id + 1
                length = len(self.tokenizer(r, add_special_tokens=False).input_ids)
                list_word = [str(entAndRel2id[r])] * length
                replace_r = ' '.join(list_word)
                length = len(self.tokenizer(t, add_special_tokens=False).input_ids)
                list_word = [str(entAndRel2id[t])] * length
                replace_t = ' '.join(list_word)
                special_sub_sentence = '<image>' + ',' + ' '+replace_r + ',' + ' '+replace_t + '<ts>'
                special_triple_sentence = special_triple_sentence + special_sub_sentence
            
            for triple in text_triple:
                h,r,t = triple
                sub_sentence = h + ',' + r + ',' + t + '<ts>'
                triple_sentence = triple_sentence + sub_sentence
                if h not in entAndRel2id:
                    entAndRel2id[h] = id 
                    id = id + 1                
                if r not in entAndRel2id:
                    entAndRel2id[r] = id 
                    id = id + 1
                if t not in entAndRel2id:
                    entAndRel2id[t] = id
                    id = id + 1
                length = len(self.tokenizer(h, add_special_tokens=False).input_ids)
                list_word = [str(entAndRel2id[h])] * length
                replace_h = ' '.join(list_word)
                length = len(self.tokenizer(r, add_special_tokens=False).input_ids)
                list_word = [str(entAndRel2id[r])] * length
                replace_r = ' '.join(list_word)
                length = len(self.tokenizer(t, add_special_tokens=False).input_ids)
                list_word = [str(entAndRel2id[t])] * length
                replace_t = ' '.join(list_word)
                special_sub_sentence = ' '+replace_h + ',' + ' '+replace_r + ',' + ' '+replace_t + '<ts>'
                special_triple_sentence = special_triple_sentence + special_sub_sentence

            for i, ip in enumerate(image_path):
                pic = self.transform(Image.open(ip).convert("RGB"))
                if i == 0:
                    pic_whole = pic.unsqueeze(0)
                else:
                    pic_whole = torch.cat((pic_whole, pic.unsqueeze(0)),0)                
        except KeyError:  # A couple of the images have no content.
            triple_sentence = ""  
        

        encoding = self.tokenizer.encode_plus(
            triple_sentence + '</s>' + caption + '</s>' + tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        decoding = self.tokenizer.encode_plus(
            sentiment_target + " is <mask>.",
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        visible_matrix = self.generate_visible_matrix(triple_sentence + '</s>' + caption + '</s>' + tweet, triple_sentence, special_triple_sentence, self.tokenizer, self.max_len)

        return {
            "review_text": tweet,
            "sentiment_targets": sentiment_target,
            "caption": caption,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": self.labels[item],
            "decoder_input_ids": decoding["input_ids"].flatten(),
            "decoder_attention_mask": decoding["attention_mask"].flatten(),
            "image_pixels": pic_whole.flatten(),
            "text": triple_sentence + '</s>' + caption + '</s>' + tweet,
            "visible_matrix": visible_matrix.flatten(),
            "encoder_text": triple_sentence,
            "decoder_text": sentiment_target + " is <mask>."
        }

    def generate_visible_matrix(self, text_whole, triple_sentence, special_triple_sentence, tokenizer, max_len):
        # step1
        tagMatrixForTripleSentence = torch.ones(len(self.tokenizer(triple_sentence).input_ids)) * -1
        triple_sentence_inputid = self.tokenizer(special_triple_sentence).input_ids
        split_index = list()
        for i,inputId in enumerate(triple_sentence_inputid):
            if inputId in [0,2,50274]:
                split_index.append(i)
            elif inputId in [50265, 6]:
                tagMatrixForTripleSentence[i] = 0
            else:
                tagMatrixForTripleSentence[i] = inputId
        # step 2 
        triple_matrix = (tagMatrixForTripleSentence.unsqueeze(0) == tagMatrixForTripleSentence.unsqueeze(1))
        for i in range(len(tagMatrixForTripleSentence)):
            for j in range(len(tagMatrixForTripleSentence)):
                if tagMatrixForTripleSentence[i] <=0 or tagMatrixForTripleSentence[j]<=0:
                    triple_matrix[i][j] = False

        for i in range(len(split_index)-1):
            triple_matrix[split_index[i]+1: split_index[i+1]+1, split_index[i]+1: split_index[i+1]+1] = True
        
        # step3
        encoding_whole = tokenizer.encode_plus(
            text_whole,
            add_special_tokens=True,
            return_tensors="pt",
        )
        whole_sentence_length = len(encoding_whole.input_ids[0])
        visible_matrix_withoutpadding = torch.ones((whole_sentence_length, whole_sentence_length))
        visible_matrix_withoutpadding[0:triple_matrix.shape[0], 0:triple_matrix.shape[1]] = triple_matrix
        # <s> is visible 
        visible_matrix_withoutpadding[0,:] = torch.ones((whole_sentence_length))
        visible_matrix_withoutpadding[:,0] = torch.ones((whole_sentence_length))

        # step4 extend the matrix to max length
        visible_matrix = torch.zeros((max_len, max_len))
        visible_matrix[0:visible_matrix_withoutpadding.shape[0], 0:visible_matrix_withoutpadding.shape[1]] = visible_matrix_withoutpadding
        return visible_matrix

