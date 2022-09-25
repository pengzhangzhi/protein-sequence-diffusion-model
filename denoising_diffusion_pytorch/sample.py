from denoising_diffusion_pytorch.pl_train import SeqDiffusion
import torch

BS = 100
ckpt_path = '/user/pengzhangzhi/personal/diffusion/denoising-diffusion-pytorch/denoising_diffusion_pytorch/experiment/best-v1.ckpt'

model = SeqDiffusion.load_from_checkpoint(
    ckpt_path,
    'cuda:1'
)
# seq_tokens = model.decode(model.repr_base.unsqueeze(0))
# print(model.idx_tns2AA(seq_tokens))

seqs = model.sample(BS)
i = 0
with open('generated_protein_seqs.fasta', 'w') as f:
    for seq in seqs:
        print(seq)
        i += 1
        f.write(">generated-" + str(i) + "\n")   
        f.write(seq + '\n')
        