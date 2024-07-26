import os
import cv2
import torch
import albumentations as A
import os
import pandas as pd


import config as CFG


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )



def download_dataset():
    os.environ['KAGGLE_USERNAME'] = ""
    os.environ['KAGGLE_KEY'] = ""

    # For Flickr 8k
    # !kaggle datasets download - d adityajn105 / flickr8k
    # !unzip flickr8k.zip
    # dataset = "8k"

    # For Flickr 30k
    # !kaggle datasets download -d hsankesara/flickr-image-dataset
    # !unzip flickr-image-dataset.zip
    # dataset = "30k"

def process_dataset(dataset):
    if dataset == "8k":
        df = pd.read_csv(f"{CFG.captions_path}/captions.txt") #pd.read_csv("captions.txt")
        df['id'] = [id_ for id_ in range(df.shape[0] // 5) for _ in range(5)]
        df.to_csv(f"{CFG.captions_path}/captions.csv", index=False)
        # df = pd.read_csv("captions.csv")
        # image_path = "/content/Images"
        # captions_path = "/content"
    # elif dataset == "30k":
    #     df = pd.read_csv("/content/flickr30k_images/results.csv", delimiter="|")
    #     df.columns = ['image', 'caption_number', 'caption']
    #     df['caption'] = df['caption'].str.lstrip()
    #     df['caption_number'] = df['caption_number'].str.lstrip()
    #     df.loc[19999, 'caption_number'] = "4"
    #     df.loc[19999, 'caption'] = "A dog runs across the grass ."
    #     ids = [id_ for id_ in range(len(df) // 5) for _ in range(5)]
    #     df['id'] = ids
    #     df.to_csv("captions.csv", index=False)
    #     image_path = "/content/flickr30k_images/flickr30k_images"
    #     captions_path = "/content"

    # df.head()


if __name__ == "__main__":
    # download_dataset()
    process_dataset("8k")
