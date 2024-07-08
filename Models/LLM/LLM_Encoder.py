import gc
import torch
from tqdm.autonotebook import trange
from Models.LLM.LLM_Model import LLMModel
from utils.utils import get_available_devices


class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="cache_data/model", batch_size=1, multi_gpu=False):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = LLMModel(llm_name, quantization=False, peft=False, cache_dir=cache_dir)
        self.model.to(self.device)

    def encode(self, 
               texts, 
               to_tensor=True
               ):
        """
        texts需要以列表的形式输入，每个元素是一个文本句子
        """
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                # 进行token化
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).to(self.device)
                # 将token转为潜空间中的向量
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()