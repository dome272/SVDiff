import collections
import os
import json
import torch
import random
import inspect
import accelerate
from pathlib import Path
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from accelerate.utils import set_module_tensor_to_device
from diffusers import UNet2DConditionModel
from safetensors.torch import safe_open
from transformers import CLIPTextModel, CLIPTextConfig


class SVDiff(nn.Module):
    def __init__(self, module, param_name, scale=1.0):
        super().__init__()
        w = getattr(module, param_name)
        self.w_shape = w.shape
        U, S, Vh = torch.linalg.svd(w.detach().view(w.size(0), -1), full_matrices=False)
        self.register_buffer("U", U, persistent=False)
        self.register_buffer("S", S, persistent=False)
        self.register_buffer("Vh", Vh, persistent=False)
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.register_buffer("scale", torch.tensor(scale, device=w.device))

    def extra_repr(self):
        return f"Scale: {self.scale}"

    def forward(self, w):
        w_alt = self.U @ torch.diag(nn.functional.relu(self.S + self.scale * self.delta)) @ self.Vh
        return w_alt.view(*self.w_shape)


def apply_svdiff(module, param_name="weight", scale=1):
    nn.utils.parametrize.register_parametrization(module, param_name, SVDiff(module, param_name, scale))
    getattr(module.parametrizations, param_name).original.data = torch.empty(0)
    return module


def convert_to_svdiff(model):
    learnable_parameters = nn.ParameterList()
    learnable_parameters_1d = nn.ParameterList()
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            apply_svdiff(module, "weight")
            learnable_parameters.append(module.parametrizations.weight[0].delta)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
            apply_svdiff(module, "weight")
            learnable_parameters_1d.append(module.parametrizations.weight[0].delta)
        elif isinstance(module, nn.MultiheadAttention):
            apply_svdiff(module, "in_proj_weight")
            learnable_parameters.append(module.parametrizations.in_proj_weight[0].delta)
    return learnable_parameters, learnable_parameters_1d

def set_scale(model, scale):
    for name, buffer in model.named_buffers():
        if "parametrizations" in name and "scale" in name:
            # print(buffer)
            buffer.data = torch.ones_like(buffer) * scale

def get_deltas(module):
    parameter_dict = {}
    for name, parameter in module.named_parameters():
        if "parametrizations" in name and "delta" in name:
            parameter_dict[name] = parameter.data
    return collections.OrderedDict(parameter_dict)


def freeze_parameters(parameters, receive_gradients=False):
    """
    Takes in model parameters, iterates through them and either freezes them or unfreezes them.
    """
    for p in parameters:
        p.requires_grad = receive_gradients


def embed_clip(caption, tokenizer, text_encoder, device):
    clip_tokens = tokenizer(caption, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
    clip_text_embeddings = text_encoder(**clip_tokens).last_hidden_state
    return clip_text_embeddings

def load_unet_and_text_encoder(unet, text_encoder, spectral_shifts_key):
    load_delta_weights(unet, spectral_shifts_key, key="unet")
    load_delta_weights(text_encoder, spectral_shifts_key, key="text_encoder", trained_token_embeds=True)

def load_delta_weights(model, spectral_shifts_ckpt, key, trained_token_embeds=False):
    if os.path.exists(spectral_shifts_ckpt):
        checkpoint = torch.load(spectral_shifts_ckpt)[key]
        model.load_state_dict(checkpoint, strict=False)
        if trained_token_embeds:
            trained_token_embeds = {}
            for name, weight in checkpoint.items():
                if name.startswith("modifier_tokens"):
                    print(f"Loading token {name}.")
                    trained_token_embeds[name.split(".")[1]] = weight
                    continue
        print(f"Resumed from {spectral_shifts_ckpt}.")
    else:
        print(f"Could not load {spectral_shifts_ckpt}, because it doesn't exist.")
    

def load_unet_for_svdiff(model, spectral_shifts_ckpt=None):
    learnable_parameters, learnable_parameters_1d = convert_to_svdiff(model)
    if spectral_shifts_ckpt:
        if os.path.exists(spectral_shifts_ckpt):
            checkpoint = torch.load(spectral_shifts_ckpt)["unet"]
            model.load_state_dict(checkpoint, strict=False)
    
    return {"params": learnable_parameters, "params_1d": learnable_parameters_1d}


def load_text_encoder_for_svdiff(
        model,
        spectral_shifts_ckpt=None,
        **kwargs
):
    learnable_parameters, learnable_parameters_1d = convert_to_svdiff(model)
    # load pre-trained weights
    trained_token_embeds = None
    if spectral_shifts_ckpt:
        if os.path.exists(spectral_shifts_ckpt):
            checkpoint = torch.load(spectral_shifts_ckpt)["text_encoder"]
            model.load_state_dict(checkpoint, strict=False)
            trained_token_embeds = {}
            for name, weight in checkpoint.items():
                if name.startswith("modifier_tokens"):
                    print(f"Adding {name} to trained_token_embeds")
                    trained_token_embeds[name.split(".")[1]] = weight
                    continue
            print(f"Resumed from {spectral_shifts_ckpt}")

    return {"params": learnable_parameters, "params_1d": learnable_parameters_1d}, trained_token_embeds


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, concepts_list, num_class_images=100, size=512, center_crop=False):
        self.size = size
        self.center_crop = center_crop

        self.instance_images_path = []
        self.class_images_path = []
        for concept in concepts_list:
            # inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.lower().endswith("jpg") or x.lower().endswith("png")]
            inst_img_path = [(os.path.join(concept["instance_data_dir"], x), concept["instance_prompt"]) for x in os.listdir(concept["instance_data_dir"]) if x.lower().endswith("jpg") or x.lower().endswith("png")]
            self.instance_images_path.extend(inst_img_path)

            class_images_path = [os.path.join(concept["class_data_dir"], "images", x) for x in os.listdir(os.path.join(concept["class_data_dir"], "images")) if x.lower().endswith("jpg") or x.lower().endswith("png")]
            # class_images_path = list(Path(os.path.join(concept["class_data_dir"], "images")).iterdir())
            # class_images_path = [file for file in class_image_path if file.lower().endswith("jpg") or file.lower().endswith("png")]
            # class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
            class_prompt = open(os.path.join(concept["class_data_dir"], "captions.txt")).read().split("\n")

            class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
            self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, antialias=False),
            transforms.RandomCrop(size),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt"] = instance_prompt

        class_image, class_prompt = self.class_images_path[index % self.num_class_images]
        class_image = Image.open(class_image)
        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        example["class_images"] = self.image_transforms(class_image)
        example["class_prompt"] = class_prompt

        return example


def collate_fn(examples, with_prior_preservation=False):
    prompts = [example["instance_prompt"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        prompts += [example["class_prompt"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return pixel_values, prompts

