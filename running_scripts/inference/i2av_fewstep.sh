export CUDA_VISIBLE_DEVICES=0

python ovi_fewstep_batch.py \
    --config_path configs/ovi_smallcfg.yaml \
    --checkpoint_path /root/dq/OviDMD/model_ema.pt \
    --csv examples/ti2av.csv \
    --output_dir outputs