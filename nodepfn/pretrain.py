import time
import argparse
from datetime import datetime

from nodepfn.scripts.model_configs import *
from nodepfn.scripts.transformer_prediction_interface import *
from nodepfn.scripts.model_builder import get_model, save_model

# from utils import init_dist
import os


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name', type=str, default='pfn')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume_epoch', type=int, default=None, help='Resume training from this epoch checkpoint')

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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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
    config['recompute_attn'] = True

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

        
    config['aggregate_k_gradients'] = 8
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

    config_sample = evaluate_hypers(config)

    model = train_function(config_sample, add_name=model_name, resume_epoch=args.resume_epoch)