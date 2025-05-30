import comfy.utils
import comfy.model_base
import comfy.model_management

import folder_paths

import torch
import os


def save_checkpoint_custom(output_path, model, clip=None, vae=None, clip_vision=None, save_precision='float16', metadata=None, extra_keys={}):
    clip_sd = None
    load_models = [model]
    if clip is not None:
        load_models.append(clip.load_model())
        clip_sd = clip.get_sd()
    vae_sd = None
    if vae is not None:
        vae_sd = vae.get_sd()

    comfy.model_management.load_models_gpu(load_models, force_patch_weights=True)
    clip_vision_sd = clip_vision.get_sd() if clip_vision is not None else None
    sd = model.model.state_dict_for_saving(clip_sd, vae_sd, clip_vision_sd)
    for k in extra_keys:
        sd[k] = extra_keys[k]

    for k in sd:
        t = sd[k]
        if not t.is_contiguous():
            sd[k] = t.contiguous()
    
    match save_precision:
        case 'float16':
            sd = comfy.utils.convert_sd_to(sd, torch.float16)
        case 'bfloat16':
            sd = comfy.utils.convert_sd_to(sd, torch.bfloat16)
        case 'float32':
            sd = comfy.utils.convert_sd_to(sd, torch.float32)
        case 'no change':
            sd = sd
        case _:
            sd = comfy.utils.convert_sd_to(sd, torch.float16)
    
    comfy.utils.save_torch_file(sd, output_path, metadata=metadata)


class CheckpointSaveCustom:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP",),
                              "vae": ("VAE",),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),
                              "filename_prefix": ("STRING", {"default": "checkpoints/ComfyUI"}),
                              "architecture": ("STRING", {"default": "stable-diffusion-xl-v1-base"}),
                              "title": ("STRING", {"default": "Custom model"}),
                              "author": ("STRING", {"default": "Anonymous"}),
                              "description": ("STRING", {"default": "A custom model"}),
                              "date": ("STRING", {"default": "2025-01-01"}),
                              "license": ("STRING", {"default": "Fair AI Public License 1.0-SD"}),
                              "predict_key": (['eps', 'v_pred', 'x0'], ),
                              "ztSNR": ("BOOLEAN", ),
                              "save_precision": (['float16', 'bfloat16', 'float32', 'no change'], ),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    CATEGORY = "advanced/model_merging"

    def save(self, model, clip, vae, filename_prefix, architecture, title, author, description, date, license, predict_key, ztSNR, save_precision, prompt=None, extra_pnginfo=None):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        
        metadata = {}

        metadata["modelspec.sai_model_spec"] = "1.0.0"
        metadata["modelspec.architecture"] = architecture
        metadata["modelspec.implementation"] = "sgm"
        metadata["modelspec.title"] = title
        
        metadata["modelspec.author"] = author
        metadata["modelspec.description"] = description
        metadata["modelspec.date"] = date

        metadata["modelspec.license"] = license

        extra_keys = {}
        extra_keys[predict_key] = torch.tensor([])
        if ztSNR:
            extra_keys['ztsnr'] = torch.tensor([])
        
        model_sampling = model.get_model_object("model_sampling")
        if isinstance(model_sampling, comfy.model_sampling.ModelSamplingContinuousEDM):
            metadata["modelspec.implementation"] = "edm"
            if isinstance(model_sampling, comfy.model_sampling.V_PREDICTION):
                extra_keys["edm_vpred.sigma_max"] = torch.tensor(model_sampling.sigma_max).float()
                extra_keys["edm_vpred.sigma_min"] = torch.tensor(model_sampling.sigma_min).float()

        if model.model.model_type == comfy.model_base.ModelType.EPS:
            metadata["modelspec.predict_key"] = "epsilon"
        elif model.model.model_type == comfy.model_base.ModelType.V_PREDICTION:
            metadata["modelspec.predict_key"] = "v"

        output_checkpoint = f"{filename}_{counter:05}_.safetensors"
        output_checkpoint = os.path.join(full_output_folder, output_checkpoint)
        
        save_checkpoint_custom(output_checkpoint, model, clip, vae, clip_vision=None, save_precision=save_precision, metadata=metadata, extra_keys=extra_keys)
        return {}


NODE_CLASS_MAPPINGS = {
    "CheckpointSaveCustom": CheckpointSaveCustom,
}
