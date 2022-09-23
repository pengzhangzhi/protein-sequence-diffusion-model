import torch
import esm
import os
from tqdm import tqdm
from typing import Iterable, List, Optional, Sequence, Tuple
import string
import string
import random
import logging
import itertools
from collections import defaultdict
import pickle
import h5py
import numpy as np
from Bio import SeqIO

def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions

def load_seq_data(fas_dpath,max_msa_depth=1000):
    """ load msa sequence from path, return a batched tensor. """
    a3m_fpath = os.path.join(fas_dpath, 'seqs.a3m')
    convert_fas_dir_2_a3m(fas_dpath,a3m_fpath,max_msa_depth)
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
   
    # load all the sequences within the msa file.
    seq_tns = parse_a3m_file(a3m_fpath,alphabet, max_msa_depth, is_train=False)
    seq_tns = torch.tensor(seq_tns)
    return seq_tns

def convert_fas_dir_2_a3m(fas_dir, a3m_path, max_seq_depth=1000):
    """ gather fas files in the fas_dir to the a3m_path.
        max_seq_depth is the maximum number of sequences in the msa file.
    """
    if os.path.exists(a3m_path) and os.path.getsize(a3m_path) > 0:
        print(f"{a3m_path} already exists.")
        return
    fas_lst = []
    for i,fas in enumerate(tqdm(os.listdir(fas_dir))):
        fas_path = os.path.join(fas_dir, fas)
        with open(fas_path) as f:
            fas_content = f.read().splitlines()
            fas_lst.append(fas_content)
        # expand 2d list to 1d
        if i > max_seq_depth:
            break
    fas_lst = [item for sublist in fas_lst for item in sublist]
    
    # write fas_lst to a3m_path
    with open(a3m_path, "w") as f:
        f.write("\n".join(fas_lst))
    return a3m_path
    
def parse_a3m_file(path, alphabet, msa_depth, is_train=False):
    """Parse the A3M file."""

    # === ESM pre-processing - BELOW ===
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None
    translation = str.maketrans(deletekeys)

    def read_sequence(filename: str) -> Tuple[str, str]:
        """Reads the first (reference) sequences from a fasta or MSA file."""
        record = next(SeqIO.parse(filename, "fasta"))
        return record.description, str(record.seq)

    def remove_insertions(sequence: str) -> str:
        """Removes any insertions into the sequence. Needed to load aligned sequences in an MSA."""
        return sequence.translate(translation)

    def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
        """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
        return [(record.description, remove_insertions(str(record.seq)))
                for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

    def read_msa_v2(filename: str, nseq: int, mode: str) -> List[Tuple[str, str]]:
        """Read sequences from an MSA file, automatically removes insertions - v2."""
        records_raw = list(SeqIO.parse(filename, "fasta"))
        records = [(x.description, remove_insertions(str(x.seq))) for x in records_raw]
        if len(records) <= nseq:
            return records
        elif mode == 'topk':
            return records[:nseq]
        else:  # then <mode> must be 'sample'
            return [records[0]] + random.sample(records, nseq - 1)

    # === ESM pre-processing - ABOVE ===
    # parse the A3M file
    converter = alphabet.get_batch_converter()
    msa_data = read_msa_v2(path, msa_depth, mode=('sample' if is_train else 'topk'))
    _, _, msa_tokens = converter(msa_data)
    msa_tokens_true = msa_tokens.squeeze(0).data.cpu().numpy()[:, 1:]

    return msa_tokens_true



if __name__ == "__main__":
    fas_dpath = "/user/pengzhangzhi/personal/diffusion/denoising-diffusion-pytorch/seq_data/fas/"
    seq_tns = load_seq_data(fas_dpath)
    print(seq_tns.shape)