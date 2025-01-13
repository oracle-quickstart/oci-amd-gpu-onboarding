# example running
ACCELERATE_USE_FSDP=1  torchrun --nproc_per_node=8 ./lora.py --config llama_3_70b_fsdp_lora.yaml

