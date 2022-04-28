
# Automatically search for available gpus
gpu_memory=(`nvidia-smi -q -d Memory |grep -A4 GPU|grep Free  | awk -F" "    '{ print $3 }'`)

subversion=5
num_gpus=1

count=0

gtx1080=10240
gtx3090=20480

available_gpus=""

batch_size=2

for i in "${!gpu_memory[@]}";   
do   
    if [ "${gpu_memory[$i]}" -gt "$gtx1080" ]
    then
        available_gpus="$available_gpus$i,"
        let count+=1
    fi
    
    if [ "${gpu_memory[$i]}" -gt "$gtx3090" ]
    then
        batch_size=8
    fi

    if [ $count -ge $num_gpus ] 
    then
        break
    fi 
done  

if [ $count -lt $num_gpus ]
then 
    echo "Error: No enough GPUs!"
    exit
fi

echo "Use GPUs: "$available_gpus

CUDA_VISIBLE_DEVICES=$available_gpus python -m torch.distributed.launch \
    --master_port 6666 \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    main.py \
    -run_type predict \
    -ckpt ./paraphrased_$subversion/ckpt-epoch10 \
    -output preds \
    -batch_size_per_gpu 8 \
    -batch_size_per_gpu_eval 32 \
    -skip_when_predict 1 \
    -subversion $subversion \
> predict_paraphrased_$subversion.log 2>&1 &
