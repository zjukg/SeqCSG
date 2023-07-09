from tokenize import triple_quoted
import pandas as pd
import json
import argparse
import logging
import os
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from dataloader import TwitterDataset
from torch.utils.data import Dataset, DataLoader

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Models.',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--MAX_LEN', type=int, default=300)
    parser.add_argument("--BATCH_SIZE", type=int, default=8)
    parser.add_argument("--DROPOUT_PROB", type=float, default=0.1)
    parser.add_argument("--NUM_CLASSES", type=int, default=3)
    parser.add_argument('--EPOCHS', type=int, default=30)
    parser.add_argument('--LEARNING_RATE', type=float, default=1e-5)
    parser.add_argument('--ADAMW_CORRECT_BIAS', type=bool, default=True)
    parser.add_argument('--NUM_WARMUP_STEPS', type=int, default=0)
    parser.add_argument('--NUM_RUNS', type=int, default=10)
    parser.add_argument('--RANDOM_SEEDS', default=42, type=int)
    parser.add_argument('--image_feature', default="resnet50", type=str)
    parser.add_argument('--dataset', default="twitter2015", type=str)
    parser.add_argument('--triple_number', default=5, type=int)
    
    args = parser.parse_args(args)
    return args


class Log(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(message)s',
                                      "%Y-%m-%d %H:%M:%S")

        # file handler
        log_file = os.path.join(log_dir, name + '.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # console handler
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

        fh.close()
        sh.close()

    def get_logger(self):
        return self.logger


def load_data(args):
    train_tsv = "/data/IJCAI2019_data/twitter2015/train.tsv".replace('twitter2015', args.dataset)
    dev_tsv = "data/IJCAI2019_data/twitter2015/dev.tsv".replace('twitter2015', args.dataset)
    test_tsv = "data/IJCAI2019_data/twitter2015/test.tsv".replace('twitter2015', args.dataset)
    image_path = "data/IJCAI2019_data/twitter2015_images".replace('twitter2015', args.dataset)
    captions_json = "data/IJCAI2019_data/captions/twitter2015_images.json".replace('twitter2015', args.dataset)
    imageid2triple_json =  "cache/twitter2015_imageid2triple.json".replace('twitter2015', args.dataset)
    
    # Load and massage the dataframes.
    test_df = pd.read_csv(test_tsv, sep="\t")
    train_df = pd.read_csv(train_tsv, sep="\t")
    val_df = pd.read_csv(dev_tsv, sep="\t")

    test_df = test_df.rename(
        {
            "index": "sentiment",
            "#1 ImageID": "image_id",
            "#2 String": "tweet_content",
            "#2 String.1": "target",
        },
        axis=1,
    )
    train_df = train_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#3 String.1": "target",
        },
        axis=1,
    ).drop(["index"], axis=1)
    val_df = val_df.rename(
        {
            "#1 Label": "sentiment",
            "#2 ImageID": "image_id",
            "#3 String": "tweet_content",
            "#3 String.1": "target",
        },
        axis=1,
    ).drop(["index"], axis=1)

    # Load the image captions.
    with open(captions_json, "r") as f:
        image_captions = json.load(f)

    with open(imageid2triple_json, "r") as f:
        imageid2triple = json.load(f)
    
    return train_df, val_df, test_df, image_captions, imageid2triple

# Construct the data loaders.
def create_data_loader(df, tokenizer, max_len, batch_size, image_captions, imageid2triple, dataset, triple_number):
    ds = TwitterDataset(
        tweets=df.tweet_content.to_numpy(),
        labels=df.sentiment.to_numpy(),
        sentiment_targets=df.target.to_numpy(),
        image_ids=df.image_id.to_numpy(),
        image_captions=image_captions,
        imageid2triple = imageid2triple,
        tokenizer=tokenizer,
        max_len=max_len,
        dataset = dataset,
        triple_number = triple_number,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=1)


def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples, tokenizer):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, total=len(data_loader)):
        input_ids = d["input_ids"].cuda()
        attention_mask = d["attention_mask"].cuda()
        decoder_input_ids = d["decoder_input_ids"].cuda()
        decoder_attention_mask = d["decoder_attention_mask"].cuda()
        targets = d["targets"].cuda()
        image_pixels = d["image_pixels"].cuda()
        visible_matrix = d["visible_matrix"].cuda()

        # experiment 1
        # output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=targets)
        # experiment 2
        # output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=targets, image_pixels=image_pixels)
        # experiment 3
        output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=targets, image_pixels=image_pixels, extended_attention_mask = visible_matrix)

        loss = output.loss
        logits = output.logits
        preds = logits.argmax(dim=1)
        correct_predictions += torch.sum(preds == targets).item()

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions / n_examples, np.mean(losses)


def format_eval_output(rows):
    tweets, targets, labels, predictions = zip(*rows)
    tweets = np.vstack(tweets)
    targets = np.vstack(targets)
    labels = np.vstack(labels)

    predictions = tuple(t.cpu() for t in predictions)
    predictions = np.vstack(predictions)
    results_df = pd.DataFrame()
    results_df["tweet"] = tweets.reshape(-1).tolist()
    results_df["target"] = targets.reshape(-1).tolist()
    results_df["label"] = labels.reshape(-1).tolist()
    results_df["prediction"] = predictions.reshape(-1).tolist()
    return results_df


def eval_model(model, data_loader, loss_fn, n_examples, detailed_results=False, tokenizer=None):
    model = model.eval()

    losses = []
    correct_predictions = 0
    rows = []

    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            input_ids = d["input_ids"].cuda()
            attention_mask = d["attention_mask"].cuda()
            decoder_input_ids = d["decoder_input_ids"].cuda()
            decoder_attention_mask = d["decoder_attention_mask"].cuda()
            targets = d["targets"].cuda()
            image_pixels = d["image_pixels"].cuda()
            visible_matrix = d["visible_matrix"].cuda()

            # experiment 1
            # output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=targets)
            # experiment 2
            # output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=targets, image_pixels=image_pixels)
            # experiment 3
            output = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids = decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=targets, image_pixels=image_pixels, extended_attention_mask = visible_matrix)

            loss = output.loss
            logits = output.logits
            preds = logits.argmax(dim=1)
            correct_predictions += torch.sum(preds == targets).item()

            losses.append(loss.item())
            rows.extend(
                zip(
                    d["review_text"],
                    d["sentiment_targets"],
                    d["targets"],
                    preds,
                )
            )

        if detailed_results:
            return (
                correct_predictions / n_examples,
                np.mean(losses),
                format_eval_output(rows),
            )

    return correct_predictions / n_examples, np.mean(losses)

