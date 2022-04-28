#3090
subversion=4
num_gpus=2

python -m torch.distributed.launch \
    --master_port 8888 \
    --nproc_per_node=$num_gpus \
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    -version 2.0 \
    -num_gpus $num_gpus \
    -run_type train \
    -batch_size_per_gpu 2 \
    -batch_size_per_gpu_eval 64 \
    -model_dir paraphrased_$subversion \
    -epochs 10 \
    -seed 42 \
    -subversion $subversion \
    -learning_rate 1e-3 \
#> train_paraphrased_$subversion.log 2>&1 &