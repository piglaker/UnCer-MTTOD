subversion=1

CUDA_VISIBLE_DEVICES=5 nohup python main.py -run_type predict -ckpt ./paraphrased_$subversion/ckpt-epoch10 -output preds -batch_size 128 -skip_when_predict 1 -subversion $subversion\
> predict_paraphrased_$subversion.log 2<&1 &
