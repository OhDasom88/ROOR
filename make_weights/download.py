from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
# bart-large
from transformers import BartTokenizer, BartModel


from tqdm import tqdm
model_paths = [
    'microsoft/layoutlmv3-large',
    'microsoft/layoutlmv3-base',
    'google-bert/bert-base-uncased',
    'google-bert/bert-large-uncased',
    'facebook/bart-base',
    'facebook/bart-large'
]
pbar = tqdm(total=len(model_paths))

# layoutlmv3-large
pbar.set_description("Downloading layoutlmv3-large")

model = AutoModel.from_pretrained('microsoft/layoutlmv3-large')
tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv3-large')

pbar.set_description("Saving layoutlmv3-large")
model.save_pretrained("microsoft/layoutlmv3-large")
tokenizer.save_pretrained("microsoft/layoutlmv3-large")
pbar.update(1)

# layoutlmv3-base
pbar.set_description("Downloading layoutlmv3-base")
model = AutoModel.from_pretrained('microsoft/layoutlmv3-base')
tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlmv3-base')

pbar.set_description("Saving layoutlmv3-base")
model.save_pretrained("microsoft/layoutlmv3-base")
tokenizer.save_pretrained("microsoft/layoutlmv3-base")
pbar.update(1)


# bert-base-uncased
pbar.set_description("Downloading bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

pbar.set_description("Saving bert-base-uncased")
model.save_pretrained("google-bert/bert-base-uncased")
tokenizer.save_pretrained("google-bert/bert-base-uncased")
pbar.update(1)

# bert-large-uncased
pbar.set_description("Downloading bert-large-uncased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-large-uncased")

pbar.set_description("Saving bert-large-uncased")
model.save_pretrained("google-bert/bert-large-uncased")
tokenizer.save_pretrained("google-bert/bert-large-uncased")
pbar.update(1)


# bart-base
pbar.set_description("Downloading bart-base")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModel.from_pretrained("facebook/bart-base")

pbar.set_description("Saving bart-base")
model.save_pretrained("facebook/bart-base")
tokenizer.save_pretrained("facebook/bart-base")
pbar.update(1)

# bart-large
pbar.set_description("Downloading bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModel.from_pretrained("facebook/bart-large")

pbar.set_description("Saving bart-large")
model.save_pretrained("facebook/bart-large")
tokenizer.save_pretrained("facebook/bart-large")
pbar.update(1)