import time
import argparse
from datetime import datetime

from scripts.model_configs import *
from scripts.transformer_prediction_interface import *
from scripts.model_builder import get_model, save_model

# from utils import init_dist
import os


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name', type=str, default='pfn')
parser.add_argument('--prior', type=str, default='geo', choices=['geo', 'causal'],
                    help="graph prior: 'geo' = casual_graph_generation similarity prior, "
                         "'causal' = original MLP + SBM/random prior bag")
parser.add_argument('--geo_similarity', type=str, default=None,
                    choices=['cosine', 'bilinear', 'mlp'],
                    help='pin the geo prior similarity kernel (default: sample it per graph)')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume_epoch', type=int, default=None, help='Resume training from this epoch checkpoint')
parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default='NodePFN', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (team/user)')
parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb run name')
# Speed/throughput overrides (default None => use the hard-coded config values).
parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
parser.add_argument('--num_steps', type=int, default=None, help='override optimizer steps per epoch')
parser.add_argument('--batch_size', type=int, default=None, help='override (nominal) batch size')
parser.add_argument('--aggregate_k_gradients', type=int, default=None,
                    help='override gradient accumulation; 1 means a full real batch per step')
parser.add_argument('--recompute_attn', dest='recompute_attn', action='store_true', default=None,
                    help='enable attention gradient checkpointing (slower, less memory)')
parser.add_argument('--no_recompute_attn', dest='recompute_attn', action='store_false',
                    help='disable attention gradient checkpointing (faster, more memory)')

args = parser.parse_args()

large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000

device = 'cuda'
base_path = '.'
max_features = 100

# using_dist, rank, device = init_dist(device)

if args.model_name is None:  
    model_name = f"test"
else: 
    model_name = args.model_name
print(model_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_models(model_string):
    print(model_string)

    for i in range(80):
        for e in range(50):
            exists = Path(os.path.join(base_path, f'models_ckpts/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.ckpt')).is_file()
            if exists:
                print(os.path.join(base_path, f'models_ckpts/prior_diff_real_checkpoint{model_string}_n_{i}_epoch_{e}.ckpt'))
        print()

def train_function(config_sample, add_name='', resume_epoch=None):
    start_time = time.time()
    N_epochs_to_save = 10
    maximum_runtime = 30
    save_dir = os.path.join(base_path, f'models_ckpts/{add_name}')
    os.makedirs(save_dir, exist_ok=True)  # exist_ok avoids a race between DDP ranks

    # If resuming, load checkpoint
    state_dict = None
    start_epoch = 0
    if resume_epoch is not None:
        checkpoint_path = os.path.join(base_path, f'models_ckpts/{add_name}/checkpoint_epoch_{resume_epoch}.ckpt')
        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            loaded = torch.load(checkpoint_path, map_location=device)
            if isinstance(loaded, tuple):
                state_dict = loaded[0]
            else:
                state_dict = loaded['model_state_dict']
            start_epoch = resume_epoch + 1
        else:
            print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")

    def save_callback(model, epoch, epochs):
        print(f'Saving model at epoch {epoch}...')
        config_sample['epoch_in_training'] = epoch
        save_model(model, base_path, f'models_ckpts/{add_name}/checkpoint_epoch_{epoch}.ckpt', config_sample)

    # Pass state_dict and start_epoch to get_model if supported, else handle in training loop
    model = get_model(
        config_sample,
        device,
        should_train=True,
        verbose=1,
        state_dict=state_dict,
        epoch_callback=save_callback,
        # start_epoch=start_epoch
    )
    # If get_model does not support start_epoch, user should handle in their training loop
    return model


def reload_config(config_type='causal', task_type='multiclass', longer=0):
    config = get_prior_config(config_type=config_type)
    
    config['prior_type'], config['differentiable'], config['flexible'] = 'prior_bag', True, True
    
    model_string = ''
    
    config['epochs'] = 20
    config['recompute_attn'] = False  # gradient checkpointing off: faster backward, uses more memory

    config['max_features'] = max_features
    config['max_num_classes'] = 20
    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False
    model_string = model_string + '_multiclass'
    
    model_string = model_string + '_'+datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    
    return config, model_string

if __name__ == "__main__":
    config, model_string = reload_config(longer=1)

    config['bptt_extra_samples'] = None
    config['output_multiclass_ordered_p'] = 0.
    del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

    config['multiclass_type'] = 'rank'
    del config['differentiable_hyperparameters']['multiclass_type']

    config['sampling'] = 'mixed' # vielleicht schlecht?
    del config['differentiable_hyperparameters']['sampling']

    config['pre_sample_causes'] = True
    config['multiclass_loss_type'] = 'nono' # 'compatible'
    config['normalize_to_ranking'] = False # False

    config['categorical_feature_p'] = .2 # diff: .0

    config['nan_prob_no_reason'] = .0
    config['nan_prob_unknown_reason'] = .0 # diff: .0
    config['set_value_to_nan'] = .1 # diff: 1.

    config['new_mlp_per_example'] = True
    config['prior_mlp_scale_weights_sqrt'] = True
    config['batch_size_per_gp_sample'] = None

    config['normalize_ignore_label_too'] = True

    config['differentiable_hps_as_style'] = False
    config['max_eval_pos'] = 1000

    config['random_feature_rotation'] = True
    config['rotate_normalized_labels'] = True

    config["mix_activations"] = True # False heisst eig True

    config['emsize'] = 512

    config['nhead'] = config['emsize'] // 128
    config['bptt'] = 1024 # 128
    config['canonical_y_encoder'] = False

        
    config['aggregate_k_gradients'] = 1  # was 8: real micro-batch == batch_size (no bs=1 pathology)
    config['batch_size'] = 8 # 64*config['aggregate_k_gradients']
    config['batch_size'] = 8 #  262144  # 64*config['aggregate_k_gradients']
    config['num_steps'] = 1024 # //config['aggregate_k_gradients']
    config['epochs'] = 30
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...
    
    config['train_mixed_precision'] = True
    config['efficient_eval_masking'] = True

    config['max_features'] = max_features
    config['max_num_classes'] = 20

    config['pos_encoder'] = 'none'

    config['use_wandb'] = args.wandb
    config['wandb_project'] = args.wandb_project
    config['wandb_entity'] = args.wandb_entity
    config['wandb_run_name'] = args.wandb_run_name if args.wandb_run_name is not None else model_name

    # Optional command-line overrides for speed/throughput experiments.
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.num_steps is not None:
        config['num_steps'] = args.num_steps
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.aggregate_k_gradients is not None:
        config['aggregate_k_gradients'] = args.aggregate_k_gradients
    if args.recompute_attn is not None:
        config['recompute_attn'] = args.recompute_attn
    print(f"[pretrain] epochs={config['epochs']} num_steps={config['num_steps']} "
          f"batch_size={config['batch_size']} aggregate_k_gradients={config['aggregate_k_gradients']} "
          f"recompute_attn={config['recompute_attn']}")

    # Select the graph prior. 'geo' replaces the MLP + SBM/random prior bag with the
    # casual_graph_generation similarity prior (features, labels and topology from one SCM).
    if args.prior == 'geo':
        config['prior_type'] = 'geo_similarity'
        config['differentiable'] = False   # geo samples its own hyperparameters internally
        config['flexible'] = False         # geo does its own normalisation / label handling
        if args.geo_similarity is not None:
            config['geo_fixed_hparams'] = {'similarity': args.geo_similarity}
    print(f"[pretrain] prior={args.prior} prior_type={config['prior_type']}")

    config_sample = evaluate_hypers(config)

    model = train_function(config_sample, add_name=model_name, resume_epoch=args.resume_epoch)