import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True)

    parser.add_argument('--logging_root', type=str, default='./output')
    parser.add_argument('--data_root', type=str, default='/om2/user/egger/MultiClassSRN/data/NMR_Dataset', required=False)
    parser.add_argument('--val_root', type=str, default=None, required=False)
    parser.add_argument('--network', type=str, default='relu')
    parser.add_argument('--category', type=str, default='donut')
    parser.add_argument('--conditioning', type=str, default='hyper')
    parser.add_argument('--experiment_name', type=str, default='experiment')
    parser.add_argument('--num_context', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_trgt', type=int, default=1)
    parser.add_argument('--views', type=int, default=2)
    parser.add_argument('--gpus', type=int, default=2)

    # General training options
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--l2_coeff', type=float, default=0.05)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--reconstruct', action='store_true', default=False)
    parser.add_argument('--lpips', action='store_true', default=False)
    parser.add_argument('--depth', action='store_true', default=False)
    parser.add_argument('--contra', action='store_true', default=False)
    parser.add_argument('--model', type=str, default='query')
    parser.add_argument('--epochs_til_ckpt', type=int, default=10)
    parser.add_argument('--steps_til_summary', type=int, default=500)
    parser.add_argument('--iters_til_ckpt', type=int, default=10000)
    parser.add_argument('--checkpoint_path', default=None)
    # Ablations
    parser.add_argument('--no_multiview', action='store_true', default=False)
    parser.add_argument('--no_sample', action='store_true', default=False)
    parser.add_argument('--no_latent_concat', action='store_true', default=False)
    parser.add_argument('--no_data_aug', action='store_true', default=False)
    parser.add_argument('--no_high_freq', action='store_true', default=False)

    return parser.parse_args()