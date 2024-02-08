# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList

MODEL_NAME = "mlabonne/phixtral-2x2_8"
MODEL_CACHE = "checkpoints"
    
class CustomStopCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores):
        # Define the IDs of tokens where the generation should stop
        self.stop_ids = [50256, 50295]
        # Check if the last generated token is one of the stop tokens
        return input_ids[0, -1].item() in self.stop_ids
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        # Enable faster download speed
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        torch.set_default_device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            trust_remote_code=True,
            load_in_8bit=True,
            cache_dir=MODEL_CACHE
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        print("setup took: ", time.time() - start)

    
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        max_new_tokens: int = Input(
            description="Max new tokens", ge=0, le=2048, default=1024
        ),
        top_p: float = Input(description="Top p", ge=0, le=1, default=0.95),
        top_k: int = Input(description="Top k", ge=0, le=100, default=50),
        temperature: float = Input(description="Temperature", ge=0, le=1, default=0.7),
        seed: int = Input(description="The seed for the random number generator", default=None),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""     
        if seed == None:
            seed = torch.randint(0, 2**30, (1,)).item()
        torch.random.manual_seed(seed)
        print("Using seed:", seed)

        system_prompt = "<|im_start|>system\nYou are Phixtral, a helpful AI assistant.<|im_end|>"
        messages = system_prompt + "\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer([messages], return_tensors="pt").to('cuda')
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        custom_stop_criteria = CustomStopCriteria()
        generate_kwargs = dict(
            input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([custom_stop_criteria])
        )
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)
        t.start()
        for new_token in streamer:
            if '<|im_end|>' in new_token:
                break
            yield new_token
        t.join()
