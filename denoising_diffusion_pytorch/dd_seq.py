import torch, torch.nn as nn
import esm
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion

class SeqDiffusion(GaussianDiffusion):
    def __init__(self):
        super().__init__()
        ...
    def tokenize(self, chars,):
        """ convert sequence of charactors to tokens (int). """
        ...
        
    def encode(self, embedding_layer:nn.Module, seqs:torch.LongTensor)-> torch.FloatTensor:
        """ encode a batch of seqs to representations.
        
            Args:
                embedding_layer: nn.Layer
                seqs: torch.LongTensor, shape = (batch_size, seq_len)
            Returns:
                representations: torch.FloatTensor, shape = (batch_size, seq_len, embedding_dim)
        """
        assert len(seqs.shape) == 2, 'seqs should be 2D tensor'
        assert isintance(seqs,torch.LongTensor), 'seqs must be a long tensor'
        repr = embedding_layer(seqs)
        assert len(repr.shape) == 3, 'repr should be 3D tensor'
        return repr

    def decode(self, representations:torch.FloatTensor, repr_base:torch.FloatTensor)-> torch.LongTensor:
        """ decode representations to seqs.
            for each repr vector, we decode it to the token that has the highest cos similarity with it. 
            
            Args:
                representations: torch.FloatTensor, shape = (batch_size, seq_len, embedding_dim)
                repr_base: torch.FloatTensor, shape = (Num of tokens, embedding_dim)
            
            Returns:
                seqs: torch.LongTensor, shape = (batch_size, seq_len)
        """
        assert len(representations.shape) == 3, 'representations should be 3D tensor'
        assert len(repr_base.shape) == 2, 'repr_base should be 2D tensor'
        bs, seq_len, _, no_tokens = *representations.shape, repr_base.shape[0]
        
        cos_sim_scores = cos_sim(representations, repr_base)
        # sample index by the probaility of cos_sim_scores 
        cos_sim_scores = cos_sim_scores.reshape(-1,no_tokens)
        cos_sim_scores = torch.softmax(cos_sim_scores,dim=-1)
        seqs = torch.multinomial(cos_sim_scores, 1).squeeze(-1)
        seqs = seqs.reshape(bs,seq_len)
        assert len(seqs.shape) == 2, 'seqs should be 2D tensor'
        return seqs
    
    def idx2AA(self,idx,mapping):
        """ convert idx tensor to amino acid sequences. 
            Args:
                idx: torch.LongTensor, shape = (batch_size, seq_len)
                mapping: list or dict, mapping idx to amino acid
            Returns:
                seqs: list of str
        """
        assert len(idx.shape) == 2, 'idx should be 2D tensor'
        assert isinstance(mapping,dict) or isinstance(mapping,list), 'mapping should be a dict or list, bud got {}'.format(type(mapping))
        seqs = []
        for i in range(idx.shape[0]):
            seq = [mapping[j] for j in idx[i]]
            seqs.append("".join(seq))
        return seqs
    
    
class SeqLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    def forward(self, seq_tokens):
        out = self.lm(seq_tokens)
        return out
    
    
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
    cos_sim = torch.matmul(tns, repr_base.T) / (tns.norm(dim=-1).unsqueeze(-1) * repr_base.norm(dim=-1))
    
    assert cos_sim.shape[-1] == repr_base.shape[0]
    return cos_sim