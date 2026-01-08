from lib.annotation.import_files import *
from vllm import LLM, SamplingParams # vllm 임포트를 이 블록 안으로 옮깁니다.


# https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Prompt_Engineering_with_Llama_3.ipynb
class VLLM:
    def __init__(self, llm_model):  
        conf_for_llm = conf.VLLM_CONF[llm_model]
        self.llm = LLM( model                   =   conf_for_llm['model'],
                        tensor_parallel_size    =   conf_for_llm['tensor_parallel_size'],   # or 4, since you have 4 GPUs
                        dtype                   =   conf_for_llm['dtype'],
                        gpu_memory_utilization  =   conf_for_llm['gpu_memory_utilization'],
                        # max_model_len           =   conf_for_llm['max_length']
                        )
        self.params = SamplingParams(temperature=   conf_for_llm['params']['temperature'], 
                                    top_p       =   conf_for_llm['params']['top_p'], 
                                    max_tokens  =   conf_for_llm['params']['max_tokens'],
                                    # stop        = ["</Difficulty Level>"]
                                    )


    


               
                
