from swift.deploy.base import Deploy
from swift.llm import MODEL_MAPPING
from swift.utils.import_utils import is_vllm_available
from swift.utils.utils import run_command_in_subprocess, close_loop, find_free_port


class Vllm(Deploy):

    def check_requirements(self):
        return is_vllm_available()

    @property
    def example(self):
        model_info = MODEL_MAPPING.get(self.model_type)
        return {
            'completion': f'curl http://localhost:{self.port}/v1/completions \\'
                          '-H "Content-Type: application/json" \\'
                          '-d \'{'
                          f'"model": "{self.checkpoint_path}",'
                          '"prompt": "San Francisco is a",'
                          '"max_tokens": 7,'
                          '"temperature": 0'
                          '}\'',
            'chat': f'curl http://localhost:{self.port}/v1/chat/completions \\'
                    '-H "Content-Type: application/json" \\'
                    '-d \'{'
                    f'"model": "{self.checkpoint_path}",'
                    '"messages": ['
                    f'{"role": "system", "content": "{model_info.get("")}"},'
                    '{"role": "user", "content": "Who won the world series in 2020?"}'
                    ']'
                    '}\''
        }

    def run_command(self):
        self.port = find_free_port()
        self.handler, logs = run_command_in_subprocess('python -m vllm.entrypoints.openai.api_server '
                                                       f'--model {self.checkpoint_path} --port {self.port}', timeout=5)
        return logs

    def close_command(self):
        close_loop(self.handler)
