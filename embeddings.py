import os
import gc
import pickle
import gzip
import argparse
import numpy as np
import pandas as pd

from scipy.special import softmax

import torch
import esm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained

from tqdm import tqdm

#create the parser
parser = argparse.ArgumentParser(
    description='Load a mutation data from output_mutation directory to compute the embeddings'
)
parser.add_argument('--name', '-n', type=str, help='The name of your csv file (without extension and path)')
#parser.add_argument('--out', '-o', type=str, help='The name of your output file')
args = parser.parse_args()

#load the data
print(f"LOAD THE DATA: {args.name}.csv")
out_dir = 'input_mutation/'
df = pd.read_csv(os.path.join(out_dir, f'{args.name}.csv'))
print('#####################################################################################################################')

#Load the model
print('LOAD THE MODEL...')
model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
#model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
#model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.eval()
if torch.cuda.is_available():
    model = model.cuda()
batch_converter = alphabet.get_batch_converter()
print('#####################################################################################################################')

#save the results
print('COMPUTE THE EMBEDDINGS...')
esm_results = {}
for i, row in tqdm(df.iterrows()):
    esm_result = {}
    data = [('protein', row.mt_seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.cuda()
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36])

    logits = (results['logits'].detach().cpu().numpy()[0,].T)[4:24,1:-1]
    #SAVE MUTATION PROBABILITIES AND ENTROPY
    esm_result['all_probs'] = softmax(logits,axis=0)
    results = results["representations"][36].detach().cpu().numpy()
    #SAVE LOCAL MUTATION EMBEDDINGS
    esm_result['all_local_embeds'] = results[0, row.mt_pos, :]
    esm_results[i] = esm_result

    del batch_tokens, results
    gc.collect(); torch.cuda.empty_cache()
print('#####################################################################################################################')

print('SAVE THE EMBEDDINGS...')
#save the esm_results into gz file
with gzip.open(f'output_embeddings/{args.name}.pkl.gz', 'wb') as f:
    pickle.dump(esm_results, f)
