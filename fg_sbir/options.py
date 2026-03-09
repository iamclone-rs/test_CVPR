import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='Fine-grained zero-shot SBIR')

    parser.add_argument('--exp_name', type=str, default='fg_prompt')
    parser.add_argument('--data_dir', type=str, default='/kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg')
    parser.add_argument('--max_size', type=int, default=224)
    parser.add_argument('--dataset', type=str, default='sketchy_fg')

    parser.add_argument('--clip_LN_lr', type=float, default=1e-5)
    parser.add_argument('--prompt_lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--patch_shuffle_margin', type=float, default=0.3)
    parser.add_argument('--lambda_cls', type=float, default=0.5)
    parser.add_argument('--lambda_patch', type=float, default=1.0)

    parser.add_argument('--prompt_dim', type=int, default=768)
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--patch_grid', type=int, default=2)

    return parser


opts = build_parser().parse_args()
