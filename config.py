import torch

debug = False
image_path = "flickr8k/Images"
captions_path = "flickr8k"
batch_size = 16
num_workers = 0
lr = 1e-3
head_lr = 1e-3
image_encoder_lr = 1e-4
text_encoder_lr = 1e-5
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256
dropout = 0.1
