import torch
from pathlib import Path
from model import RNN, text_pipeline
from preprocess import preprocess
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

vocab_size = 100000
num_embd = 256
rnn_hidden = 128
fcl_hidden = 64

model = RNN(vocab_size, num_embd, rnn_hidden, fcl_hidden)
model.load_state_dict(torch.load(f'{BASE_DIR}/model_taghche-{__version__}.pth', weights_only=True))

def predict_pipeline(text):
    model.eval()
    text = preprocess(text)
    length = len(text)
    
    transformed = text_pipeline(text)
    transformed = torch.tensor(transformed).unsqueeze(0)
    length = torch.tensor(length).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(transformed, length)
    
    return pred.view(-1)