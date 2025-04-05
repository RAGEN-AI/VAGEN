#!/bin/bash

set -e
set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

python -m vagen.env.eb_alfred.create_dataset \
    --data_dir data/alfred-fast-debug \
    --start_seed 0 \
    --train_ratio 0.8 \
    --n_candidate 20 \
    --force-gen \
    --resolution 300 \
    --eval_set base \
    --exp_name fast_debug \
    --down_sample_ratio 1.0\
    --max_action_per_step 1 \
    --max_action_penalty -0.1 \
    --format_reward 0.5

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/alfred-fast-debug/train.parquet \
    data.val_files=data/alfred-fast-debug/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=64 \
    data.max_trajectory_length=768 \
    data.image_key=images \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=[wandb,console] \
    trainer.project_name='vagen-fast-debug' \
    trainer.experiment_name='debug_alfred_fast' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 \
    rollout_manager.max_turns=2 \
    rollout_manager.window_size=2 \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=0 \
    rollout_manager.n_trajectory=1 \
    2>&1 | tee debug_alfred_fast.log

