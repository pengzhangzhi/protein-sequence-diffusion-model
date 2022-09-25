import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import pairwise2 as pw2
from Bio import SeqIO 
import numpy as np

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)
    
def l2_norm_loss(pred, true):
    pred_normed = l2_norm_tns(pred)
    # no necessary, if true is x_0. because is aready normalized. otherwise if it is gussain noise, this step is necessary.
    true_normed = l2_norm_tns(true) 
    l2_loss = F.mse_loss(pred_normed,true_normed,reduction='none').sum(-1)
    return l2_loss


def l2_loss_with_norm_penalty(pred,true):
    """ 
    calculate the normalized l2 loss and the norm penalty.
    The implementation is similar to the Algorithm 27 in Alpahfold2 supplementary material.
    this loss measures the distance the angle between the predicted repr and ground truth angle.
    """
    l2_loss = l2_norm_loss(pred, true)
    norm_penalty = torch.abs(pred.norm(dim=-1,p=2)-1)
    loss = l2_loss + 0.02 * norm_penalty
    return loss

def l2norm(t):
    return F.normalize(t, dim = -1)

def l2_norm_tns(tns):
    """ l2 normalize the tns """
    return tns / tns.norm(dim=-1,p=2,keepdim=True)

def l2_distance(reprs,repr_base):
    """ 
    calculate l2 distance between reprs and repr_base.
    
    Args:
        reprs: torch.FloatTensor, shape = (*, embedding_dim)
        repr_base: torch.FloatTensor, shape = (num of tokens, embedding_dim)
    Returns:
        cos_sim: torch.FloatTensor, shape = (*,num of tokens)
            
    """
    reprs = reprs.unsqueeze(-2)
    return torch.nn.functional.mse_loss(reprs,repr_base,reduction='none').mean(-1)

def calc_entropy(x):
    b = F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)
    b = -1.0 * b.sum(-1)
    return b

def cos_sim(tns, repr_base):
    """ 
    calculate cosine similarity between reprs in tns and repr_base.
    each repr in tns will be calculated with all the reprs in repr_base to get a cos similarity score.
        Args:
            tns: torch.FloatTensor, shape = (*, embedding_dim)
            repr_base: torch.FloatTensor, shape = (num of tokens, embedding_dim)
        Returns:
            cos_sim: torch.FloatTensor, shape = (*,num of tokens)
    """
    assert tns.shape[-1] == repr_base.shape[-1], 'the last dimension of tns and repr_base should be the same'
    # cos_sim = torch.matmul(tns, repr_base.T) / (tns.norm(dim=-1).unsqueeze(-1) * repr_base.norm(dim=-1))
    dist_squre = l2_norm_loss(tns.unsqueeze(-2), repr_base)
    cos_sim_score = 1-(1/2)*dist_squre
    assert cos_sim_score.shape[-1] == repr_base.shape[0]
    return cos_sim_score

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num



def calc_pairwise_sequence_similarity(fasta_src_fpath,fasta_tgt_fpath):
    with open(fasta_src_fpath) as f1:
        first_dict = SeqIO.to_dict(SeqIO.parse(f1,'fasta')) 
    with open(fasta_tgt_fpath) as f2:
        second_dict = SeqIO.to_dict(SeqIO.parse(open(fasta_tgt_fpath),'fasta'))
    score_matrix = np.ones((len(first_dict.keys()),len(second_dict.keys())))
    # 两个fasta文件中的序列两两比较：
    for i,t in enumerate(first_dict):
        t_len = len(first_dict[t].seq)
        for j,t2 in enumerate(second_dict):
            global_align = pw2.align.globalxx(first_dict[t].seq, second_dict[t2].seq)
            matched = global_align[0][2]
            percent_match = (matched / t_len) * 100
            score_matrix[i,j] = percent_match
            
    return score_matrix
def is_letter(char):
    return char.isalpha()

def is_capitalized(char):
    return char.isupper()

def filter_lst(lst,fn):
    """ filter a list, delete items that return False by the `fn`. """
    return [item for item in lst if fn(item)]

def remove_non_aa(seq_lst):
    """ remove non-aa characters from a sequence list. """
    assert isinstance(seq_lst,list)
    return filter_lst(seq_lst,is_capitalized)


restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

def alphabet_esm_2_af2(esm_alphabet_dict):
    """ convert esm alphabet to alphafold2 alphabet.
        remove the mapping of aa characters that are not in af2 alphabet.
        these are uncommon amino acid.
        
    """
    def _convert(aa):
        if aa in restypes:
            return aa
        else:
            return ""
    return {k:_convert(v) for k,v in esm_alphabet_dict.items()}