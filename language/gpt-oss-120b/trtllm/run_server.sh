#!/bin/bash

for var in $(compgen -v | grep '^SLURM_'); do unset "$var"; done

model_path=openai/gpt-oss-120b
extra_args=""
eagle_path=""
draft_len=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            model_path=$2
            shift 2
            ;;
        --eagle_path)
            eagle_path=$2
            shift 2
            ;;
        --draft_len)
            draft_len=$2
            shift 2
            ;;
        *)
            extra_args="$extra_args $2"
            ;;
    esac
done


cat <<EOF > config.yml
enable_layerwise_nvtx_marker: false
stream_interval: 10
disable_overlap_scheduler: false
enable_iter_perf_stats: true
enable_chunked_prefill: true
scheduler_config:
  capacity_scheduler_policy: MAX_UTILIZATION
  context_chunking_policy: FIRST_COME_FIRST_SERVED
kv_cache_config:
  dtype: fp8 
  free_gpu_memory_fraction: 0.95
  enable_block_reuse: false
moe_config:
  backend: TRTLLM
cuda_graph_config:
  enable_padding: true
  max_batch_size: 512
enable_attention_dp: true
attention_dp_config:
  enable_balance: true
num_postprocess_workers: 4
print_iter_log: true
sampler_type: TorchSampler
EOF

if [ -n "$eagle_path" ]; then
    cat <<EOF >> config.yml
speculative_config:
    decoding_type: Eagle
    max_draft_len: $draft_len
    speculative_model_dir: $eagle_path
    eagle3_layers_to_capture: [-1]
EOF
fi

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

set -x;

for ((gpu=0; gpu<gpu_count; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu trtllm-serve $model_path --host 0.0.0.0 --port 3000$gpu --backend pytorch --max_batch_size 512 --max_num_tokens 4096 --tp_size 1 --ep_size 1 --trust_remote_code --extra_llm_api_options config.yml $extra_args &
done

# num_servers=2
# CUDA_VISIBLE_DEVICES=0,1,2,3 TRTLLM_ENABLE_PDL=1 trtllm-serve $model_path --host 0.0.0.0 --port 30000 --backend pytorch --max_batch_size 1024 --tp_size 4 --ep_size 1 --trust_remote_code --extra_llm_api_options config.yml $extra_args & > $output_dir/trtllm-serve-0.log 2>&1
# CUDA_VISIBLE_DEVICES=4,5,6,7 TRTLLM_ENABLE_PDL=1 trtllm-serve $model_path --host 0.0.0.0 --port 30001 --backend pytorch --max_batch_size 1024 --tp_size 4 --ep_size 1 --trust_remote_code --extra_llm_api_options config.yml $extra_args & > $output_dir/trtllm-serve-1.log 2>&1

wait
