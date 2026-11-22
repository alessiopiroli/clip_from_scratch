import json
from pathlib import Path

from tqdm import tqdm


class Preprocessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = Path(self.cfg.DATA.root_dir)
        self.train_images_names_file = Path(self.cfg.DATA.train_images_names_file)
        self.dev_images_names_file = Path(self.cfg.DATA.dev_images_names_file)
        self.test_images_names_file = Path(self.cfg.DATA.test_images_names_file)
        self.all_captions_file = Path(self.cfg.DATA.all_captions_file)
        self.tokenizer_data_file = Path(self.cfg.TOKENIZER.tokenizer_data_file)

        self.vocabulary_size = self.cfg.TOKENIZER.vocab_size
        self.splits = ["train", "dev", "test"]

    # method which finds the names of all the images of a split
    def get_images_paths(self, split):
        path = Path.joinpath(self.root_dir, f"Flickr_8k.{split}Images.txt")

        if not path.exists():
            raise FileNotFoundError(f"Could not open {path}")

        with open(path, "r") as f:
            filenames = [line.rstrip("\n") for line in f]

        return filenames

    # method which returns, for each image of the split, the list of its captions
    # image_name: [caption0, caption1, ...]
    def get_captions_by_image(self, split_images_paths):
        path = Path.joinpath(self.root_dir, self.all_captions_file)

        if not path.exists():
            raise FileNotFoundError(f"Could not open {path}")

        split_images_paths = set(split_images_paths)
        captions_by_image = {}

        with open(path, "r") as f:
            for line in f:
                image_and_caption_num, caption = line.rstrip("\n").split("\t")
                filename, _ = image_and_caption_num.split("#")
                if filename in split_images_paths:
                    if filename not in captions_by_image:
                        captions_by_image[filename] = [caption]
                    else:
                        captions_by_image[filename].append(caption)

        assert len(captions_by_image) == len(split_images_paths)
        for captions in captions_by_image.values():
            assert len(captions) == 5

        return captions_by_image

    # method which returns all captions of a split as a single string, one caption per line
    def get_captions(self, split_images_paths):
        captions_by_image = self.get_captions_by_image(split_images_paths)

        training_captions = []

        for captions in captions_by_image.values():
            for caption in captions:
                training_captions.append(caption + "\n")

        training_captions = "".join(training_captions)

        return training_captions

    # method which updates the tokens
    @staticmethod
    def update_tokens(tokens, most_frequent):
        new_tokens = []
        i = 0

        while i < len(tokens):
            if (i + 1 < len(tokens)) and ((tokens[i], tokens[i + 1]) == most_frequent):
                new_tokens.append("".join(most_frequent))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1

        return new_tokens

    # method which returns the frequencies of tokens
    @staticmethod
    def get_frequencies(tokens):
        frequencies = {}
        for pair in zip(tokens, tokens[1:]):
            if pair not in frequencies:
                frequencies[pair] = 1
            else:
                frequencies[pair] += 1

        return frequencies

    # method which applies bpe and returns the vocabulary and merges
    def bpe(self, text, vocabulary_size):
        tokens = list(text)
        vocabulary = set(tokens)
        merges = []

        for _ in tqdm(range(vocabulary_size - len(vocabulary)), desc=f"Creating vocabulary and merges"):
            frequencies = self.get_frequencies(tokens)

            if not frequencies:
                break

            # max returns the first pair with the highest frequency, same as sorting and taking the first
            most_frequent = max(frequencies, key=frequencies.get)
            tokens = self.update_tokens(tokens, most_frequent)
            vocabulary.add("".join(most_frequent))
            merges.append(most_frequent)

        return vocabulary, merges

    # method which saves to json file the vocabulary and the merges
    def save_tokenizer(self, vocabulary, merges):
        tokenizer_data = {"vocabulary": list(vocabulary), "merges": merges}

        tokenizer_data_file = Path.joinpath(self.root_dir, self.tokenizer_data_file)

        with open(tokenizer_data_file, "w") as f:
            json.dump(tokenizer_data, f)

    # method which creates the vocabulary, the merges and then saves them to json file
    def create_vocab_and_merges(self, training_captions):
        vocabulary, merges = self.bpe(training_captions, self.vocabulary_size)
        self.save_tokenizer(vocabulary, merges)

    # method which applied one merge to a string
    @staticmethod
    def do_one_merge(text, merge):
        new_text = []
        i = 0
        while i < len(text):
            if (i + 1 < len(text)) and (merge == (text[i], text[i + 1])):
                new_text.append("".join(merge))
                i += 2
            else:
                new_text.append(text[i])
                i += 1

        return new_text

    # method which tokenizes one caption
    def tokenize_one_caption(self, caption):
        text = list(caption)
        # pairs currently in the caption, a merge is applied only if its pair is present
        pairs = set(zip(text, text[1:]))
        for merge in self.merges:
            if merge in pairs:
                text = self.do_one_merge(text, merge)
                pairs = set(zip(text, text[1:]))

        return text

    # method which creates the json file used for a split
    # image_name: [caption0, caption1, ...], each caption is a list of tokens
    def create_split_inputs(self, split):
        split_images_paths = self.get_images_paths(split)
        captions_by_image = self.get_captions_by_image(split_images_paths)

        split_data = {}

        for image_name in tqdm(split_images_paths, desc=f"Tokenizing {split} captions"):
            tokenized_captions = []
            for caption in captions_by_image[image_name]:
                tokenized_captions.append(self.tokenize_one_caption(caption))
            split_data[image_name] = tokenized_captions

        split_data_file = Path.joinpath(self.root_dir, f"{split}_data.json")

        with open(split_data_file, "w") as f:
            json.dump(split_data, f)

    # method which loads the tokenzer data file
    def load_tokenizer_data_file(self):
        tokenizer_data_file = Path.joinpath(self.root_dir, self.tokenizer_data_file)

        if not tokenizer_data_file.exists():
            raise FileNotFoundError(f"Could not file at {tokenizer_data_file}")

        with open(tokenizer_data_file, "r") as f:
            tokenizer_data = json.load(f)

        return tokenizer_data

    # method which creates the json files used in the dataset class
    def create_dataset_inputs(self):
        self.tokenizer_data = self.load_tokenizer_data_file()
        self.merges = [tuple(merge) for merge in self.tokenizer_data["merges"]]

        for split in self.splits:
            self.create_split_inputs(split)

    # main method that does the preprocessing of the data for the dataset class
    def preprocess(self):
        train_images_paths = self.get_images_paths("train")
        training_captions = self.get_captions(train_images_paths)
        self.create_vocab_and_merges(training_captions)
        self.create_dataset_inputs()
