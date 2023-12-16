import json
import os
from abc import abstractmethod, ABC

from swift import snapshot_download
from swift.llm import merge_lora, InferArguments, MODEL_MAPPING
from swift.utils import get_logger


logger = get_logger()


class Deploy(ABC):

    TORCH = 'torch'
    VLLM = 'vllm'

    def __init__(self, checkpoint_path_or_id, **kwargs):
        """
        Args:
            checkpoint_path_or_id: The checkpoint path of swift or a original model path or id
        """
        self.checkpoint_path, self.sft_args = self.read_config(checkpoint_path_or_id)

    @staticmethod
    def read_config(checkpoint_path_or_id):
        if not os.path.exists(checkpoint_path_or_id):
            checkpoint_path_or_id = snapshot_download(checkpoint_path_or_id)
        with open(os.path.join(checkpoint_path_or_id, 'sft_args.json'), 'r') as f:
            return checkpoint_path_or_id, json.load(f)

    @property
    def sft_type(self):
        return self.sft_args.get('sft_type')

    @property
    def model_type(self):
        return self.sft_args.get('model_type')

    @staticmethod
    def merge_lora(checkpoint_path_or_id):
        if not os.path.exists(checkpoint_path_or_id):
            checkpoint_path_or_id = snapshot_download(checkpoint_path_or_id)
        return merge_lora(InferArguments(ckpt_dir=checkpoint_path_or_id))

    @classmethod
    def merge(cls, sft_type, checkpoint_path):
        if sft_type in ('lora', 'longlora', 'qalora'):
            return cls.merge_lora(checkpoint_path)
        elif sft_type == 'full':
            return checkpoint_path
        else:
            raise ValueError(f'Merge not supported: {sft_type}')

    @abstractmethod
    def check_requirements(self):
        pass

    @abstractmethod
    def run_command(self):
        pass

    @abstractmethod
    def close_command(self):
        pass

    @abstractmethod
    @property
    def example(self):
        pass

    @classmethod
    def deploy(cls, checkpoint_path_or_id, deploy_type=None, **kwargs):
        checkpoint_path, sft_args = cls.read_config(checkpoint_path_or_id)
        sft_type = sft_args.get('sft_type')
        model_type = sft_args.get('model_type')
        model_info = MODEL_MAPPING.get(model_type)
        deploy_type = deploy_type if deploy_type is not None else model_info.get('deploy_type')
        if deploy_type is None:
            raise ValueError(f'Please pass in a deploy type')

        if deploy_type == cls.VLLM:
            from swift.deploy.vllm_singleton import Vllm
            checkpoint_path = cls.merge(sft_type, checkpoint_path)
            deployer = Vllm(checkpoint_path, **kwargs)
        elif deploy_type == cls.TORCH:
            from swift.deploy.vllm_singleton import Torch
            checkpoint_path = cls.merge(sft_type, checkpoint_path)
            deployer = Torch(checkpoint_path, **kwargs)
        else:
            raise ValueError(f'Deploy type not supported: {deploy_type}')

        logger.info(deployer.run_command())
        return deployer

