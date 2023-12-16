from swift.deploy.base import Deploy
from swift.deploy.protocol import ChatCompletionRequest
from swift.llm import InferArguments
from swift.utils.import_utils import is_fastapi_available
from swift.llm.infer import inference, inference_stream, prepare_model_template


class Torch(Deploy):

    def __init__(self, checkpoint_path_or_id, **kwargs):
        self.kwargs = kwargs
        super().__init__(checkpoint_path_or_id)

    def check_requirements(self):
        return is_fastapi_available()

    @property
    def example(self):
        pass

    def run_command(self):
        from fastapi import FastAPI
        app = FastAPI()
        args = InferArguments(**self.kwargs)
        model, template = prepare_model_template(args)
        if args.overwrite_generation_config:
            assert args.ckpt_dir is not None
            model.generation_config.save_pretrained(args.ckpt_dir)

        @app.post("/v1/chat/completions")
        async def create_chat_completion(request: ChatCompletionRequest):
            if not isinstance(request.messages, str):
                query = request.messages[-1]['']
            gen = inference_stream(model, template, query, history)
            print_idx = 0
            for response, history in gen:
                if len(response) > print_idx:
                    print(response[print_idx:], end='', flush=True)
                    print_idx = len(response)
            print()
            print('-' * 50)
            item = history[-1]
            obj = {
                'query': item[0],
                'response': item[1],
                'history': history,
            }
            if jsonl_path is not None:
                append_to_jsonl(jsonl_path, obj)
            result.append(obj)



    def close_command(self):
        pass



