# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_prefix_indices
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os

from megatron.text_generation_utils import generate_and_write_samples_unconditional
try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:  ##True
            args.pretrain_causal_attention = True
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.causal
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=args.loss_on_targets_only
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=args.loss_on_targets_only
    )
    if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path: ## ['data/meg-gpt2-oscar-en-10k_text_document']
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(name, paths, weights, splits,
                                        args.data_impl,
                                        train_val_test_num_samples,
                                        args.seq_length, args.seed,
                                        (not args.mmap_warmup),
                                        train_valid_test=s)
                eval(f"{s}_ds").append(d)

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
    print_rank_0("finish!!!!!!!!!!!!!!!!!!!!")




#Namespace(num_layers=2, hidden_size=8, ffn_hidden_size=32, num_attention_heads=2, kv_channels=4, max_position_embeddings=512,\
#  make_vocab_size_divisible_by=128, pad_vocab_size_to=None, layernorm_epsilon=1e-05, \
# sync_tp_duplicated_parameters=False, \
# apply_residual_connection_post_layernorm=False, \
# embed_layernorm=True, openai_gelu=False, onnx_safe=None, bert_binary_head=True,\
#  position_embedding_type=<PositionEmbeddingType.absolute: 2>, \
# glu_activation=None, kill_switch_path='/tmp/kill-switch', \
# log_level=None, log_level_replica=None, attention_dropout=0.1, \
# hidden_dropout=0.1, weight_decay=0.1, clip_grad=1.0, adam_beta1=0.9, adam_beta2=0.95, \
# adam_eps=1e-08, sgd_momentum=0.9, micro_batch_size=1, global_batch_size=16,\
#  rampup_batch_size=['2', '2', '1_000'], checkpoint_activations=True, \
# distribute_checkpointed_activations=False, checkpoint_num_layers=1, \
# train_iters=None, train_samples=10000, train_tokens=None, log_interval=10, \
# exit_interval=100, exit_duration_in_mins=None, tensorboard_dir='output_dir/tensorboard',\
#  masked_softmax_fusion=True, bias_gelu_fusion=True, bias_dropout_fusion=True, \
# optimizer='adam', use_bnb_optimizer=False, dataloader_type='single', \
# cpu_optimizer=False, cpu_torch_adam=False, codecarbon_dir=None, \
# eval_only=None, skip_train_iteration_range=None, inference=False, \
# abort_on_unmet_fused_kernel_constraints=False, pp_partition_method=None, \
# seed=42, init_method_std=0.02, init_method_xavier_uniform=False, lr=0.0001, \
# lr_decay_style='cosine', lr_decay_iters=None, lr_decay_samples=12,\
#  lr_decay_tokens=None, lr_warmup_fraction=None, lr_warmup_iters=0, \
# lr_warmup_samples=5, min_lr=1e-06, override_lr_scheduler=False, \
# use_checkpoint_lr_scheduler=False, universal_checkpoint=False, \
# save='checkpoints/gpt2', save_interval=50, no_save_optim=None,\
#  no_save_rng=None, load='checkpoints/gpt2', no_load_optim=None, \
# no_load_rng=None, finetune=False, fp16=True, bf16=False, loss_scale=None, \
# initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, \
# hysteresis=2, fp32_residual_connection=False, apply_query_key_layer_scaling=True, \
# attention_softmax_in_fp32=False, accumulate_allreduce_grads_in_fp32=False,\
#  fp16_lm_cross_entropy=False, tensor_model_parallel_size=2, pipeline_model_parallel_size=1,\
#  num_layers_per_virtual_pipeline_stage=None, distributed_backend='nccl',\
#  DDP_impl='local', use_contiguous_buffers_in_ddp=False, \
# scatter_gather_tensors_in_pipeline=True, local_rank=0, lazy_mpu_init=None, \
# use_cpu_initialization=None, eval_iters=10, eval_interval=100, \
# data_path=['data/meg-gpt2-oscar-en-10k_text_document'], split='969, 30, 1', \
# train_weighted_split_paths=None, valid_weighted_split_paths=None, \
# test_weighted_split_paths=None, train_weighted_split_paths_path=None, \
# valid_weighted_split_paths_path=None, test_weighted_split_paths_path=None,\
#  log_path=None, vocab_file='data/gpt2-vocab.json', merge_file='data/gpt2-merges.txt', \
# vocab_extra_ids=0, seq_length=512, encoder_seq_length=512, decoder_seq_length=None, \
# retriever_seq_length=256, sample_rate=1.0, mask_prob=0.15, short_seq_prob=0.1,\
#  mmap_warmup=False, num_workers=2, valid_num_workers=2, tokenizer_type='GPT2BPETokenizer',\
#  tokenizer_name_or_path=None, data_impl='infer', reset_position_ids=False,\
#  reset_attention_mask=False, eod_mask_loss=False, loss_on_targets_only=False,\
# reweight_loss_based_on_position_frequency=False, noise_density=None, \
# mean_noise_span_length=None, adlr_autoresume=False, adlr_autoresume_interval=1000, \
# ict_head_size=None, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, \
# ict_load=None, bert_load=None, titles_data_path=None, query_in_block_prob=0.1, \
# use_one_sent_docs=False, evidence_data_path=None, retriever_report_topk_accuracies=[], \
# retriever_score_scaling=False, block_data_path=None, embedding_path=None, \
# indexer_batch_size=128, indexer_log_interval=1000, num_classes=1000, \
# img_dim=224, num_channels=3, patch_dim=16, log_params_norm=False,\
#  log_num_zeros_in_grad=False, tensorboard_log_interval=1, tensorboard_queue_size=5, \
# log_timers_to_tensorboard=True, log_batch_size_to_tensorboard=True, \
# log_learning_rate_to_tensorboard=True, log_loss_scale_to_tensorboard=True, \
# log_validation_ppl_to_tensorboard=True, zero_stage=1, zero_reduce_scatter=False,\
#  zero_contigious_gradients=False, zero_reduce_bucket_size=0.0, \
# zero_allgather_bucket_size=0.0, remote_device='none', use_pin_memory=False, \
# scattered_embeddings=False, split_transformers=False, memory_centric_tiled_linear=False, \
# tile_factor=1, deepspeed_activation_checkpointing=True, partition_activations=True, \
# contigious_checkpointing=False, checkpoint_in_cpu=False, synchronize_each_layer=False,\
#  profile_backward=False, deepspeed=True, deepspeed_config='./ds_config.json', \
# deepscale=False, deepscale_config=None, deepspeed_mpi=False, rank=0, world_size=2, \
# data_parallel_size=1, valid_weighted_split_names=None, valid_weighted_split_weights=None,\
#  valid_weighted_split_splits=None, test_weighted_split_names=None, \
# test_weighted_split_weights=None, test_weighted_split_splits=None,\
#  virtual_pipeline_model_parallel_size=None, params_dtype=torch.float16, \
# consumed_train_samples=0, consumed_valid_samples=0, consumed_train_tokens=0, \
# gigaflos_no_embeds=0, curriculum_learning=False, padded_vocab_size=50432, \
# deepspeed_configuration={'train_micro_batch_size_per_gpu': 1, 'train_batch_size': 16, \
# 'gradient_clipping': 1.0, 'zero_optimization': {'stage': 1}, \
# 'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 500, 'hysteresis': 2, 'min_loss_scale': 1, 'initial_scale_power': 12}, \
# 'steps_per_print': 2000, 'wall_clock_breakdown': False})