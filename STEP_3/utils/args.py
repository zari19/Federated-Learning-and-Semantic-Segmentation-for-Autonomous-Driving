import argparse

def modify_command_options(args):
    if args.dataset == 'cityscapes':
        args.num_classes = 19
    elif args.dataset == 'idda' and args.remap:
        args.num_classes = 16
    elif args.dataset == 'idda' and not args.remap:
        args.num_classes = 23
    args.total_batch_size = len(args.device_ids) * args.batch_size
    args.device_ids = [int(device_id) for device_id in args.device_ids]
    args.n_devices = len(args.device_ids)
    return args

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'femnist'], required=True, help='dataset name')
    parser.add_argument('--niid', action='store_true', default=False,
                        help='Run the experiment with the non-IID partition (IID by default). Only on FEMNIST dataset.')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--print_train_interval', type=int, default=10, help='client print train interval')
    parser.add_argument('--print_test_interval', type=int, default=10, help='client print test interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval interval')
    parser.add_argument('--test_interval', type=int, default=10, help='test interval')
    parser.add_argument("--local_rank", type=int, default = 0, help = 'rank')
    parser.add_argument('--device_ids', default=[0], nargs='+', help='GPU ids for multigpu mode')
    parser.add_argument('--load', default=False, help='Whether to use pretrained or not')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='if you want wandb offline set to True, otherwise it uploads results on cloud')
    parser.add_argument('--wandb_entity', type=str, default='feddrive', help='name of the wandb entity')
    parser.add_argument('--avg_last_100', action='store_true', default=False,
                        help='compute avg and std last 100 rounds for each test type')
    parser.add_argument('--random_seed', type=int, required=False, help='random seed')
    parser.add_argument('--remap', action='store_true', default=False, help='Whether to remap IDDA as Cityscapes or'
                                                                            'not')
    parser.add_argument('--save_samples', action='store_true', default=False, help='Save samples pictures on cloud')
    parser.add_argument('--plot', default=True, help='Save test image in test_imgs foolder')
    parser.add_argument('--ckpt', default=False, help='Save checkpoints')
    parser.add_argument('--domain', type=str, choices=['same', 'diff'], required=False, help='test same/diff dom')
    parser.add_argument('--step', type=str, choices=['1','2','3','4','5'], required=False, help='select the number of step')
    parser.add_argument('--modality4', type=str, choices=['1', '2', '3'], required=False, help='Student teacher interaction type')
   
   


    return parser
