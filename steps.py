import argparse
import os
import sys

# Define the folder paths and main file names
folders = ["/root/STEP_1", "/root/STEP_2", "/root/STEP_3", "/root/STEP_4","/root/STEP_5"]
main_files = ["main.py", "main.py", "main.py", "main.py","main.py"]

# Create a dictionary to map step values to folder indices
step_to_folder = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4
}

def execute_main(step):
    if step not in step_to_folder:
        print("Invalid step value. Step should be 1, 2, 3, 4, or 5.")
        sys.exit(1)

    folder_index = step_to_folder[step]
    folder = folders[folder_index]
    main_file = main_files[folder_index]

    # Build the command to execute the main file
    command = f"python {os.path.join(folder, main_file)} {' '.join(sys.argv[1:])}"
    os.system(command)

def main():
    parser = argparse.ArgumentParser(description="Execute main files based on step value.")
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5], help="Step value (1, 2, 3, 4, or 5)")

    # Add other command-line parameters here using argparse
    parser.add_argument("--dataset", choices=["idda", "femnist"], required=True, help="Dataset choice")
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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
    parser.add_argument('--load', action='store_true', default=False, help='Whether to use pretrained or not')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='if you want wandb offline set to True, otherwise it uploads results on cloud')
    parser.add_argument('--wandb_entity', type=str, default='feddrive', help='name of the wandb entity')
    parser.add_argument('--avg_last_100', action='store_true', default=False,
                        help='compute avg and std last 100 rounds for each test type')
    parser.add_argument('--remap', action='store_true', default=False, help='Whether to remap IDDA as Cityscapes or'
                                                                            'not')
    parser.add_argument('--save_samples', action='store_true', default=False, help='Save samples pictures on cloud')
    parser.add_argument('--plot', default=True, help='Save test image in test_imgs foolder')
    parser.add_argument('--ckpt', default=False, help='Save checkpoints')
    parser.add_argument('--domain', type=str, choices=['same', 'diff'], required=False, help='test same/diff dom')
    parser.add_argument('--modality4', type=str, choices=['1', '2', '3'], required=False,help='Student teacher interaction type')
   

    args = parser.parse_args()
    execute_main(args.step)

if __name__ == "__main__":
    main()



