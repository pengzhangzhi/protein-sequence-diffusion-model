from utils import exists

def add_train_arg(parser):
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--save_dir", type=str, default="experiment")
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--max_time_steps", type=int, default=1000)
    parser.add_argument("--token_dim", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.003)
    return parser

def add_eval_arg(parser):
    parser.add_argument("--sample_size", type=int, default=10)
    return parser
    
def add_data_arg(parser):
    """
    Note:
        fas_dpath: dir path where stores the sequence fasta files.
        max_msa_depth: maximum number of sequences for training.

    """
    parser.add_argument(
        "--fas_dpath",
        type=str,
        default="/user/pengzhangzhi/personal/diffusion/denoising-diffusion-pytorch/denoising_diffusion_pytorch/seq_data/fas",
    )
    parser.add_argument("--max_msa_depth", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=169)
    return parser

def parse_args(cli=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser = add_train_arg(parser)
    parser = add_eval_arg(parser)
    parser = add_data_arg(parser)
    args = parser.parse_args()
    if exists(cli):
        args = parser.parse_args(cli)
    return args
