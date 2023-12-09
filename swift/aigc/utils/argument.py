# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

from swift import get_logger
from swift.utils import broadcast_string, get_dist_setting, is_dist

logger = get_logger()


@dataclass
class AnimateDiffArguments:
    motion_adapter_id_or_path: Optional[str] = None
    motion_adapter_revision: Optional[str] = None

    model_id_or_path: str = None
    model_revision: str = None

    dataset_sample_size: int = None

    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})

    output_dir: str = 'output'
    ddp_backend: str = field(
        default='nccl', metadata={'choices': ['nccl', 'gloo', 'mpi', 'ccl']})

    seed: int = 42

    gradient_checkpointing: bool = False
    batch_size: int = 1
    num_train_epochs: int = 1
    # if max_steps >= 0, override num_train_epochs
    max_steps: int = -1
    learning_rate: Optional[float] = None
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 16
    max_grad_norm: float = 1.
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.05

    eval_steps: int = 50
    save_steps: Optional[int] = None
    dataloader_num_workers: int = 1

    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    push_hub_strategy: str = field(
        default='push_best',
        metadata={'choices': ['push_last', 'all_checkpoints']})
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
            'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })

    ignore_args_error: bool = False  # True: notebook compatibility

    text_dropout_rate: float = 0.1

    validation_prompts_path: str = field(
        default=None,
        metadata={
            'help':
            'The validation prompts file path, use llm/configs/ad_validation.txt is None'
        })

    trainable_modules: str = field(
        default='.*motion_modules.*',
        metadata={
            'help':
            'The trainable modules, by default, the .*motion_modules.* will be trained'
        })

    mixed_precision: bool = True

    enable_xformers_memory_efficient_attention: bool = True

    num_inference_steps: int = 25
    guidance_scale: float = 8.
    sample_size: int = 256
    sample_stride: int = 4
    sample_n_frames: int = 16

    csv_path: str = None
    video_folder: str = None

    motion_num_attention_heads: int = 8
    motion_max_seq_length: int = 32
    num_train_timesteps: int = 1000
    beta_start: int = 0.00085
    beta_end: int = 0.012
    beta_schedule: str = 'linear'
    steps_offset: int = 1
    clip_sample: bool = False

    use_wandb: bool = False

    def __post_init__(self) -> None:
        current_dir = os.path.dirname(__file__)
        if self.validation_prompts_path is None:
            self.validation_prompts_path = os.path.join(
                current_dir, 'configs/animatediff', 'validation.txt')
        if self.learning_rate is None:
            self.learning_rate = 1e-4
        if self.save_steps is None:
            self.save_steps = self.eval_steps

        if is_dist():
            rank, local_rank, _, _ = get_dist_setting()
            torch.cuda.set_device(local_rank)
            self.seed += rank  # Avoid the same dropout
            # Initialize in advance
            if not dist.is_initialized():
                dist.init_process_group(backend=self.ddp_backend)
            # Make sure to set the same output_dir when using DDP.
            self.output_dir = broadcast_string(self.output_dir)


@dataclass
class AnimateDiffInferArguments:

    motion_adapter_id_or_path: Optional[str] = None
    motion_adapter_revision: Optional[str] = None

    model_id_or_path: str = None
    model_revision: str = None

    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})

    ckpt_dir: Optional[str] = field(
        default=None, metadata={'help': '/path/to/your/vx_xxx/checkpoint-xxx'})
    eval_human: bool = False  # False: eval val_dataset

    seed: int = 42

    # other
    ignore_args_error: bool = False  # True: notebook compatibility

    validation_prompts_path: str = None

    output_path: str = './generated'

    enable_xformers_memory_efficient_attention: bool = True

    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    sample_size: int = 256
    sample_stride: int = 4
    sample_n_frames: int = 16

    motion_num_attention_heads: int = 8
    motion_max_seq_length: int = 32
    num_train_timesteps: int = 1000
    beta_start: int = 0.00085
    beta_end: int = 0.012
    beta_schedule: str = 'linear'
    steps_offset: int = 1
    clip_sample: bool = False

    def __post_init__(self) -> None:
        pass


@dataclass
class DreamBoothArguments:
    instance_data_dir: str = None
    image_column: str = None
    class_data_dir: str = None
    instance_prompt: str = None
    class_prompt: str = None
    with_prior_preservation: bool = None
    num_class_images: int = None
    repeats: int = 1
    prior_loss_weight: float = 1.0


@dataclass
class SDXLArguments:
    resolution: int = 1024
    crops_coords_top_left_h: int = 0
    crops_coords_top_left_w: int = 0
    center_crop: bool = False


@dataclass
class LoRAArguments:
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.05


@dataclass
class HubArguments:
    push_to_hub: bool = False
    # 'user_name/repo_name' or 'repo_name'
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True
    push_hub_strategy: str = field(
        default='push_best',
        metadata={'choices': ['push_last', 'all_checkpoints']})
    # None: use env var `MODELSCOPE_API_TOKEN`
    hub_token: Optional[str] = field(
        default=None,
        metadata={
            'help':
                'SDK token can be found in https://modelscope.cn/my/myaccesstoken'
        })


@dataclass
class SDXLDreamBoothArguments(SDXLArguments, DreamBoothArguments, LoRAArguments, HubArguments):
    model_id_or_path: str = None
    model_revision: str = None
    vae_model_id_or_path: str = None
    vae_revision: str = None
    seed: int = None
    variant: str = None
    dataset_name: str = None
    dataset_config_name: str = None
    validation_prompt: str = None
    num_validation_images: int = 5
    validation_epochs: int = None
    output_dir: str = None
    train_text_encoder: bool = False
    train_batch_size: int = 4
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = None
    resume_from_checkpoint: str = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    text_encoder_lr: float = 5e-6
    scale_lr: bool = False
    lr_scheduler: str = 'constant'
    snr_gamma: float = None
    lr_power: float = 1.0
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    dataloader_num_workers: int = 1
    max_train_steps: int = None
    optimizer: str = 'AdamW'
    use_8bit_adam: bool = False
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    prodigy_beta3: float = None
    adam_weight_decay: float = 1e-4
    adam_weight_decay_text_encoder: float = 1e-3
    adam_epsilon: float = 1e-8
    prodigy_use_bias_correction: bool = True
    prodigy_safeguard_warmup: bool = True
    max_grad_norm: float = 1.0
    logging_dir: str = 'logs'
    allow_tf32: bool = True
    report_to: str = 'tensorboard'
    mixed_precision: str = field(
        default='None',
        metadata={
            'choices': ["no", "fp16", "bf16"],
        }
    )
    prior_generation_precision: str = field(
        default='None',
        metadata={
            'choices': ["no", "fp32", "fp16", "bf16"],
        }
    )
    enable_xformers_memory_efficient_attention: bool = False
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})

    def __post_init__(self) -> None:
        if self.dataset_name is None and self.instance_data_dir is None:
            raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

        if self.dataset_name is not None and self.instance_data_dir is not None:
            raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

        if self.with_prior_preservation:
            if self.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if self.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
        else:
            # logger is not available yet
            if self.class_data_dir is not None:
                warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
            if self.class_prompt is not None:
                warnings.warn("You need not use --class_prompt without --with_prior_preservation.")