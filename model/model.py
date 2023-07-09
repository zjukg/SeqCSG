import os
import torch
import torch.nn as nn
import numpy as np
from transformers import T5Tokenizer
from modeling_bart import BartForSequenceClassification
from torchvision import models

class SentimentClassifier(nn.Module):
    def __init__(self, args, tokenizer):
        super(SentimentClassifier, self).__init__()
        self.args = args
        if args.image_feature == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
        elif args.image_feature == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
        elif args.image_feature == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
        elif args.image_feature == "resnet101":
            self.resnet = models.resnet101(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(self.resnet.fc.out_features, 768)

        self.bart = BartForSequenceClassification.from_pretrained("facebook/bart-base", num_labels=3)
        self.bart.resize_token_embeddings(len(tokenizer))

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels, image_pixels, extended_attention_mask=[]):
        image_pixels = image_pixels.reshape(-1,3,224,224)
        image_embedding = self.linear(self.resnet(image_pixels))
        if extended_attention_mask == []:
            outputs = self.bart(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                decoder_input_ids = decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask, 
                labels = labels, image_embedding=image_embedding, 
                extended_attention_mask = extended_attention_mask)
        else:
            extended_attention_mask = extended_attention_mask.reshape(-1, 1, self.args.MAX_LEN, self.args.MAX_LEN)
            outputs = self.bart(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                decoder_input_ids = decoder_input_ids, 
                decoder_attention_mask=decoder_attention_mask, 
                labels = labels, 
                image_embedding=image_embedding, 
                extended_attention_mask = extended_attention_mask)

        return outputs