import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class llmAbs:
    def __init__(self, model_name):
        self.check_gpu()
        self.check_memory()
        self.model_name = model_name
        self.init_model(model_name)
    
    def check_gpu(self):
        if not torch.cuda.is_available():
            logging.warning('GPU device not found. Go to Runtime > Change Runtime type and set Hardware accelerator to "GPU"')
            logging.warning('If you use CPU it will be very slow')
        else:
            print(f"Cuda device found: {torch.cuda.get_device_name(0)}")

    def check_memory(self):
        memory_info = torch.cuda.mem_get_info()
        print(f"Free Memory Usage: {memory_info[0]}\nTotal Available Memory: {memory_info[1]}")
      
    def init_model(self, model_name):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        
        # Set pad token for batched generation
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Get inputs
        self.inputs = self.get_inputs(2)
        
        # Set parameters
        self.generation_kwargs = self.set_kwargs()
        
        # Generate outputs hqq[1/2/3/4/8/fp16]
        self.selected_types = (2,)
        self.outputs = dict()
        for type in self.selected_types:
            self.outputs[type] = self.generate_output(type)
            if self.outputs[type] is None:
                raise Exception("Generated output cannot be NoneType. Invalid selected type(s).")
        
        # Display results
        self.display()
        
        # Terminate cuda
        self.terminate()
    
    def get_inputs(self, num_questions=391):
        sub_prompts = []
        with open("forbidden_question_set.csv", "r") as f:
            f.readline()
            for line in f:
                line = (*line.strip().split(','), )
                if int(line[2]) < num_questions:
                    sub_prompts.append(line[3])
                else:
                    return self.tokenizer(sub_prompts, padding=True, return_tensors="pt").to(self.model.device)
    
    def set_kwargs(self):
        gk = {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 40,
            "min_new_tokens": 20
        }
        return gk
    
    def generate_output(self, n_bits):
        print(f"Generating output ({n_bits}).")
        match n_bits:
            case n if n in (1,2,3,4,8):
                return self.model.generate(**self.inputs, cache_implementation="quantized",
                    cache_config=
                    {
                        "backend": "HQQ",
                        "nbits": n_bits,
                        "q_group_size": 32,
                        "residual_length": 64
                    }
                )
            case 'fp16':
                inputs = self.inputs
                generation_kwargs = self.generation_kwargs
                return self.model.generate(**inputs, **generation_kwargs)
            case _:
                print(f"Quant/Model with bit-size {n_bits} is not supported by quanto or hqq. [1,2,3,4,8,fp16]")
    
    def display(self):
        print(f"\n\n{self.model_name}:",end="\n\n")
        for type, out in self.outputs.items():
            output_list = self.tokenizer.batch_decode(out)
            print(f"Cache {type}: {output_list}",end="\n\n")
            for response in output_list:
                print(response)
            print("\n\n")
    
    def terminate(self):
        torch.cuda.empty_cache()
        
