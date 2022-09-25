import torch, torch.nn as nn
import esm
from denoising_diffusion_pytorch import GaussianDiffusion
from denoising_diffusion_pytorch.ddpt import exists
from denoising_diffusion_pytorch.seq_data import get_loader
from denoising_diffusion_pytorch import parse_args
from torch import optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from denoising_diffusion_pytorch.utils import LambdaLayer, alphabet_esm_2_af2, calc_entropy,cos_sim, l2_distance, l2_norm_tns,remove_non_aa
from pytorch_lightning.loggers import CSVLogger
 

# pl.LightningModule
class SeqDiffusion(pl.LightningModule):
    def __init__(self, cfg=parse_args([])):
        super().__init__()
        self.lr = cfg.lr
        lm, alphabet = esm.pretrained.esm2_t6_8M_UR50D(
            max_time_steps=cfg.max_time_steps
        )
        # fix params of embedding layer.
        lm.embed_tokens.weight.requires_grad = False
        # first embed tokens to representations and norm it to unit length.
        self.token2repr = nn.Sequential(
            lm.embed_tokens, LambdaLayer(l2_norm_tns)
        )
        lm.embed_tokens.to(cfg.device)
        
        self.alphabet = alphabet
        self.diffusion_model = GaussianDiffusion(
            model=SeqLM(lm, alphabet).to(cfg.device),
            sample_shape=(cfg.max_seq_len, cfg.token_dim),
            timesteps=cfg.max_time_steps,
            loss_type="l2",
            objective = 'pred_x0',
        ).to(cfg.device)
        
        self._init_alphabet_dict(alphabet)
        self._init_repr_database(cfg)

    def _init_alphabet_dict(self, alphabet):
        # init idx to amino acid charcters
        self.idx2char = {v: k for k, v in alphabet.to_dict().items()}
        self.idx2char = alphabet_esm_2_af2(self.idx2char)
        self.num_tokens = len(self.idx2char)

    def _init_repr_database(self, cfg):
        # init representation database, contains the repr of each charcters
        with torch.no_grad():
            self.repr_base = self.token2repr(
                torch.arange(
                    0,
                    self.num_tokens,
                ).to(cfg.device)
            ).to(cfg.device)
            

    def encode(self, tokens):
        return self.token2repr(tokens)

    def decode(self, representations: torch.FloatTensor,mode='hard') -> torch.LongTensor:
        """decode representations to seqs.
        for each repr vector, we decode it to the token that has the highest cos similarity with it.

        Args:
            representations: torch.FloatTensor, shape = (batch_size, seq_len, embedding_dim)

        Returns:
            seqs: torch.LongTensor, shape = (batch_size, seq_len)
        """
        assert len(representations.shape) == 3, "representations should be 3D tensor"
        assert len(self.repr_base.shape) == 2, "repr_base should be 2D tensor"
        bs, seq_len, _, no_tokens = *representations.shape, self.repr_base.shape[0]
        cos_sim_scores = cos_sim(representations, self.repr_base.to(representations.device))
        if mode == "hard":
            seqs = cos_sim_scores.reshape(bs, seq_len, no_tokens).argmax(-1)
        elif mode == 'soft':
            cos_sim_scores = cos_sim_scores.reshape(-1, no_tokens)
            cos_sim_scores = (cos_sim_scores + 1) / 2 
            # cos_sim_scores[cos_sim_scores<cos_sim_scores.max(-1)[0]] = cos_sim_scores[cos_sim_scores<cos_sim_scores.max(-1)[0]] ** 3
            max_en = calc_entropy(torch.ones(no_tokens)/no_tokens)
            entropy = calc_entropy(cos_sim_scores)
            entropy = entropy.reshape(-1)
            en_mean, en_std = entropy.mean(), entropy.std()
            print(f"entropy mean: {en_mean}, std: {en_std}, max: {max_en}")
            seqs = torch.multinomial(cos_sim_scores, 1).squeeze(-1)
            seqs = seqs.reshape(bs, seq_len)
            assert len(seqs.shape) == 2, "seqs should be 2D tensor"
            # print(seqs)
        else:
            raise ValueError("mode should be 'hard' or 'soft'")
        return seqs
    
    def idx_tns2AA(
        self,
        idx_tns,
    ):
        """convert idx tensor to amino acid sequences.
        Args:
            idx: torch.LongTensor, shape = (batch_size, seq_len)
        Returns:
            seqs: list of str
        """
        assert len(idx_tns.shape) == 2, "idx should be 2D tensor"
        idx_tns = idx_tns.cpu().numpy()
        seqs = []
        # map numpy array to charcters with self.idx2char
        # map_fn = np.vectorize(self.idx2char.get)
        # seqs = ["".join(remove_non_aa(lst)) for lst in map_fn(idx_tns).tolist()]
        for i in range(idx_tns.shape[0]):
            seq = [self.idx2char[j] for j in idx_tns[i]]
            seq = remove_non_aa(seq)
            seqs.append("".join(seq))
        return seqs

    def sample(self, batch_size):
        """ sample `batch_size` sequences from the diffusion model. """
        print(f"sampling {batch_size} samples on {self.device}")
        sampled_repr = self.diffusion_model.sample(batch_size)
        sampled_tokens = self.decode(sampled_repr)
        aa_seqs = self.idx_tns2AA(sampled_tokens)
        return aa_seqs

    def training_step(self, batch, batch_idx):
        # pre-generate padding_mask
        padding_mask = batch.eq(self.alphabet.padding_idx)
        reprs = self.token2repr(batch)
        loss = self.diffusion_model(reprs, padding_mask=padding_mask)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.diffusion_model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
        #                                                 max_lr=self.lr,
        #                                                 steps_per_epoch=self.num_training_batches,
        #                                                 epochs=self.max_epochs,
        #                                                 anneal_strategy='linear')
        return optimizer


class SeqLM(nn.Module):
    def __init__(self, lm, alphabet, self_condition=False):
        super().__init__()
        self.lm, self.alphabet = lm, alphabet
        self.self_condition = self_condition
        assert self.self_condition == False, "WIP feature"
        self.num_layers = lm.num_layers
        # cancel embedding layer,
        # such that the lm does not take token as input
        # but take the representation of token as input.
        self.lm.embed_tokens = nn.Identity()

    def forward(self, seq_repr, t, x_self_cond, padding_mask):
        """feed seq_repr to a lm. output repr should have the same shape of input.
        Note:
            x_self_cond comes from paper Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning.
            now this feature is WIP, so it is not used.
        """
        # if not exists(padding_mask):
        #     padding_mask = torch.zeros(*seq_repr.shape[:2],dtype=torch.bool).to(seq_repr.device)
        out = self.lm(
            seq_repr, repr_layers=[self.num_layers], t=t, padding_mask=padding_mask
        )["representations"][self.num_layers]
        assert out.shape == seq_repr.shape
        return out


def train(args):
    train_loader = get_loader(args)
    args.num_training_batches = len(train_loader)
    args.max_seq_len = train_loader.dataset.max_seq_len
    denosing_model = SeqDiffusion(args)
    logger = CSVLogger(args.save_dir, name="train_loss")
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.save_dir,
        filename="best",
    )
    early_stop = EarlyStopping(
        monitor="train_loss", mode="min", patience=0.1*(args.max_epochs), verbose=True
    )
    gpu_id = int(args.device.split(":")[-1])
    trainer = pl.Trainer(
        # overfit_batches=1,
        max_epochs=args.max_epochs,
        gpus=[gpu_id],
        default_root_dir=args.save_dir,
        callbacks=[checkpoint_callback, early_stop],# early_stop
        logger=logger,
    )
    trainer.fit(model=denosing_model, train_dataloaders=train_loader)
    seqs = denosing_model.sample(args.sample_size)
    with open('generated_protein_seqs.txt', 'w') as f:
        for seq in seqs:
            print(seq)
            f.write(seq + '\n')
        

if __name__ == "__main__":
    args = parse_args()
    train(args)
