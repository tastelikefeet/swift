from .animatediff import animatediff_sft
from .animatediff_infer import animatediff_infer
from .utils.argument import AnimateDiffArguments, AnimateDiffInferArguments
from swift.aigc.diffusers.train_controlnet import main as train_controlnet
from swift.aigc.diffusers.train_controlnet_sdxl import main as train_controlnet_sdxl
from swift.aigc.diffusers.train_dreambooth import main as train_dreambooth
from swift.aigc.diffusers.train_dreambooth_lora import main as train_dreambooth_lora
from swift.aigc.diffusers.train_dreambooth_lora_sdxl import main as train_dreambooth_lora_sdxl
from swift.aigc.diffusers.train_lcm_distill_lora_sd_wds import main as train_lcm_distill_lora_sd_wds
from swift.aigc.diffusers.train_lcm_distill_sd_wds import main as train_lcm_distill_sd_wds
from swift.aigc.diffusers.train_lcm_distill_sdxl_wds import main as train_lcm_distill_sdxl_wds
from swift.aigc.diffusers.train_lcm_distill_lora_sdxl_wds import main as train_lcm_distill_lora_sdxl_wds
from swift.aigc.diffusers.train_text_to_image_sdxl import main as train_text_to_image_sdxl
from swift.aigc.diffusers.train_text_to_image_lora_sdxl import main as train_text_to_image_lora_sdxl
from swift.aigc.diffusers.train_text_to_image import main as train_text_to_image
from swift.aigc.diffusers.train_text_to_image_lora import main as train_text_to_image_lora

