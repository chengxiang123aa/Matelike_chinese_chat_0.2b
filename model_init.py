import torch
import random
import numpy as np
from transformers import AutoTokenizer
from model.model import Transformer
from model.LMConfig import LMConfig

class Chatbot:
    def __init__(self, model_path = r"model_save\full_sft_512.pth", tokenizer_path = r"tokenizer\mateconv_tokenizer", seed=1337, max_seq_len=1024, temperature=0.5, top_k=16):
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        self.setup_seed(seed)
        
        # 配置模型
        self.lm_config = LMConfig()
        self.lm_config.max_seq_len = max_seq_len
        self.model = Transformer(self.lm_config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 模型推理参数
        self.temperature = temperature
        self.top_k = top_k
        self.max_seq_len = max_seq_len
    
    def setup_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def generate_reply(self, prompt, stream=True):
        """
        生成完整回复
        
        :param prompt: 输入提示
        :param stream: 是否使用流式生成
        :return: 生成的回复
        """
        messages = [{"role": "user", "content": prompt}]
        
        # 使用自定义的 prompt 模板
        new_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )[-(self.max_seq_len - 1):]

        input_ids = self.tokenizer(new_prompt).data['input_ids']
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        generated_text = ""
        with torch.no_grad():
            # 生成器返回的生成结果
            res_y = self.model.generate(input_ids, 
                                        self.tokenizer.eos_token_id, 
                                        max_new_tokens=self.max_seq_len, 
                                        temperature=self.temperature, 
                                        top_k=self.top_k, 
                                        stream=stream)

            # 从生成器逐步获取生成结果
            try:
                y = next(res_y)
            except StopIteration:
                print("No answer")
                return ""

            history_idx = 0
            while y is not None:
                answer = self.tokenizer.decode(y[0].tolist())
                if answer and answer[-1] == '�':
                    try:
                        y = next(res_y)
                    except StopIteration:
                        break
                    continue

                if len(answer):
                    generated_text += answer[history_idx:]
                
                try:
                    y = next(res_y)
                except StopIteration:
                    break
                history_idx = len(answer)

        return generated_text


