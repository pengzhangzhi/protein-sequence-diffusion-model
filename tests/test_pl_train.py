import unittest
from denoising_diffusion_pytorch.pl_train import SeqDiffusion,cos_sim
from denoising_diffusion_pytorch import utils
import torch, torch.nn as nn 
class TestSeqDiffusion(unittest.TestCase):
    bs = 32
    no_tokens = 20
    seq_len = 100
    embedding_dim = 64
    
    tns = torch.randn(bs,seq_len,embedding_dim)
    repr_base = torch.randn(no_tokens,embedding_dim)
    idx_tns = torch.randint(0,no_tokens,(bs,seq_len))
    mapping = {i:chr(i+65) for i in range(no_tokens)}
    seqs = torch.randint(0,no_tokens,(bs,seq_len))
    embedding_layer = nn.Embedding(no_tokens,embedding_dim)
    reprs = torch.randn(bs,seq_len, embedding_dim)
    SD = SeqDiffusion()
    def test_encode(self,):
        repr = self.SD.encode(self.embedding_layer,self.seqs)
        
    def test_decode(self,):
        seqs = self.SD.decode(self.reprs,self.repr_base)
        
    def test_idx2AA(self,):
        idx = torch.ones(self.bs,self.seq_len).fill_(0).long()
        mapping = utils.restypes_with_x
        seqs = self.SD.idx2AA(idx,mapping)
        assert "".join(seqs) == 'A'*(self.bs*self.seq_len)
        
    def test_cos_sim(self,):
        score = cos_sim(self.tns,self.repr_base)
        self.assertTrue(torch.all(( score <= 1)))
        self.assertTrue(torch.all(( score >= -1)))
        
if __name__ == '__main__':
    unittest.main()