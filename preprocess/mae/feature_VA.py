import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from functools import partial
from PIL import Image
from models_mae import MaskedAutoencoderViT  # 导入 MAE 模型类


class MAEFeatureExtractor(MaskedAutoencoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        del self.decoder_embed
        del self.mask_token
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred
        del self.decoder_pos_embed

    def forward(self, imgs):
        latent, _, _ = self.forward_encoder(imgs, mask_ratio=0.0)
        cls_token_feature = latent[:, 0, :]
        return cls_token_feature
def load_pretrained_mae_feature_extractor(ckpt_path):
    model = MAEFeatureExtractor(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    encoder_state_dict = {k: v for k, v in checkpoint.items() if 'decoder' not in k}

    model_state_dict = model.state_dict()


    filtered_state_dict = {}
    for k, v in encoder_state_dict.items():
        if k in model_state_dict:
            filtered_state_dict[k] = v
        else:
            print(f"Skipping key {k} as it's not in the encoder")


    model.load_state_dict(filtered_state_dict, strict=True)
    return model


transform = transforms.Compose([
    transforms.Resize(size=(224, 224), interpolation=Image.BICUBIC, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.471, 0.363, 0.333], std=[0.220, 0.193, 0.180])
])


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        

        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('png', 'jpg', 'jpeg')):
                    self.img_paths.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path


def extract_and_save_features_batch(model, dataloader, save_dir, root_dir):
    model.eval()

    with torch.no_grad():
        for imgs, img_paths in dataloader:
            imgs = imgs.cuda()
            features = model(imgs)
            

            save_features(features.cpu().numpy(), img_paths, save_dir, root_dir)

def save_features(batch_features, batch_paths, save_dir, root_dir):

    for i, feature in enumerate(batch_features):
        img_path = batch_paths[i]

        relative_path = os.path.relpath(img_path, root_dir)
        feature_save_path = os.path.join(save_dir, os.path.splitext(relative_path)[0] + ".npy")
        

        os.makedirs(os.path.dirname(feature_save_path), exist_ok=True)
        

        np.save(feature_save_path, feature)

        print(f"Feature saved for {os.path.basename(img_path)} at {feature_save_path}")


if __name__ == "__main__":


    ckpt_path = "/data/lyh/8th_result/logs/best_model.pth"
    model = load_pretrained_mae_feature_extractor(ckpt_path)
    model.cuda()


    img_dir = '/data/lyh/8th_data/test_data/cropped_aligned_test'
    dataset = CustomImageDataset(img_dir, transform=transform)
    

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)


    save_dir = "/data/lyh/8th_result/pre_test/npy_test"
    extract_and_save_features_batch(model, dataloader, save_dir, img_dir)
