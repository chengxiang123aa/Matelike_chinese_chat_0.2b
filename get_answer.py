from model_init import Chatbot

model_path = r"model_save\full_sft_512.pth"
tokenizer_path = r"tokenizer\mateconv_tokenizer"

model = Chatbot(model_path,tokenizer_path)

prompt = '你好！好久不见'

answer = model.generate_reply(prompt)
print(answer)
