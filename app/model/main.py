import torch
from pathlib import Path
import warnings
from torch.utils.data import DataLoader

from app.model.model import RNN, collate_fn
from app.model.preprocess import preprocess

warnings.filterwarnings("ignore", category=FutureWarning)
__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

vocab_size = 100000
num_embd = 256
rnn_hidden = 128
fcl_hidden = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNN(vocab_size, num_embd, rnn_hidden, fcl_hidden)
# model.load_state_dict(torch.load(f'{BASE_DIR}/model_taghche-{__version__}.pth', weights_only=True))
model.load_state_dict(torch.load(f'{BASE_DIR}/model_taghche-{__version__}.pth', map_location=torch.device('cpu')))

model = model.to(device)

def predict_pipeline(text):
    text = preprocess(text)
    
    usage_loader = DataLoader([[text, 1]], batch_size=1, collate_fn=collate_fn)

    text, label, lenghts = next(iter(usage_loader))
    
    model.eval()
    text = text.to(device)
    lenghts = lenghts.to(device)
    with torch.no_grad():
        pred = model(text, lenghts)
    
    return pred.view(-1)