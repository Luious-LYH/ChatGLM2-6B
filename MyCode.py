from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(r"../model/chatglm2-6b", trust_remote_code=True)

model = AutoModel.from_pretrained("../model/chatglm2-6b", trust_remote_code=True, device='cuda')

model = model.eval()