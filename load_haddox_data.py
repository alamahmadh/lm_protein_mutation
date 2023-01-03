import os
import re
import collections

import warnings
warnings.simplefilter('ignore') # ignore warnings

import numpy as np
import pandas as pd
import seaborn

from Bio import Seq, SeqIO
from Bio.Seq import translate

import scipy.stats
from scipy.special import softmax 
from scipy.stats import entropy, spearmanr, rankdata

from sklearn.model_selection import train_test_split

#directory the data from Haddox et al (2018)
data_dir = 'dms_data/haddox2018/data/'
#!cp data/hiv/fitness_haddox2018/BG505_avgprefs.csv dms_data/haddox2018/data/BG505_avgprefs.csv

#consider two env proteins
homologs = [ 'BF520', 'BG505']
seqs_fitness = []
for env in homologs:
    wt_seq = translate(SeqIO.read(os.path.join(data_dir, '{}_env.fasta'.format(env)), 'fasta').seq).rstrip('*')
    fname = os.path.join(data_dir, '{}_to_HXB2.csv'.format(env))
    pos_map = {}
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            pos_map[fields[1]] = (fields[2], int(fields[0])-1)
        
    with open(os.path.join(data_dir,'{}_avgprefs.csv'.format(env))) as f:
        mutants = f.readline().rstrip().split(',')[1:]
        for line in f:
            fields = line.rstrip().split(',')
            orig, pos = pos_map[fields[0]]
            fitnesses = [float(field) for field in fields[1:]]
            for mt, fit in zip(mutants, fitnesses):
                seq_fitness = {}
                seq = [aa for aa in wt_seq]
                wt = seq[pos]
                if seq[pos] != mt:
                    seq[pos] = mt
                    mt_seq = ''.join(seq)
                    wt_seq = ''.join(wt_seq)
                    seq_fitness['env'] = env
                    seq_fitness['wt_seq'] = wt_seq
                    seq_fitness['mt_seq'] = mt_seq
                    seq_fitness['mt_pos'] = pos
                    seq_fitness['wt_aa'] = mt
                    seq_fitness['mt_aa'] = wt
                    seq_fitness['fitness'] = fit
                    seqs_fitness.append(seq_fitness)

df_fit_env = pd.DataFrame(seqs_fitness)