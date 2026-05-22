import isolated_prior.priors.differentiable_prior as differentiable_prior
from isolated_prior.config import config

if __name__ == "__main__":
    # Example usage
    steps_per_epoch = 8192
    batch_size = 1
    device = "cpu:0"
    bptt = 1024
    bptt_extra_samples = None
    from nodepfn.utils import get_uniform_single_eval_pos_sampler

    def eval_pos_seq_len_sampler():
        single_eval_pos = get_uniform_single_eval_pos_sampler(
            config["hyperparameters"].get(
                "max_eval_pos", config["hyperparameters"]["bptt"] # 1024
            ),
            min_len=config["hyperparameters"].get("min_eval_pos", 100), # 100
        )()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt

    dataloader = differentiable_prior.DifferentiablePriorDataLoader(
        num_steps=steps_per_epoch,
        batch_size=batch_size,
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler,
        seq_len_maximum=bptt,
        device=device,
        **config
    )
    
    for batch in dataloader:
        print(batch)
        break
