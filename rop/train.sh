# export TOKENIZERS_PARALLELISM=false
# export PYTHONPATH=/path/to/ROOR/rop


# CUDA_VISIBLE_DEVICES=0 python src/tasks/pl_training.py \
#   --do_train true \
#   --image_dir /path/to/ROOR \
#   --json_dir /path/to/ROOR/jsons \
#   --split_file_dir /path/to/ROOR \
#   --train_dataset_name data.train.txt \
#   --valid_dataset_name data.val.txt \
#   --bbox_level segment \
#   --unit_type segment \
#   --max_num_units 256 \
#   --pretrained_model_path /path/to/layoutlmv3-large-2048 \
#   --save_model_dir /path/to/save_model_dir \
#   --batch_size 2 \
#   --accumulate_grad_batches 8 \
#   --max_epochs 500 \
#   --learning_rate 2e-5 \
#   --log_every_n_steps 1 \
#   --keep_checkpoint_max 1 \
#   --patience 50 \
#   --shuffle true \
#   --seed 2024 \
#   --gpus 1


export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/dasom/ROOR/rop


CUDA_VISIBLE_DEVICES=0 python src/tasks/pl_training.py \
  --do_train true \
  --image_dir /home/dasom/ROOR/ROOR-Datasets/data \
  --json_dir /home/dasom/ROOR/ROOR-Datasets/data/jsons \
  --split_file_dir /home/dasom/ROOR/ROOR-Datasets/data \
  --train_dataset_name data.train.txt \
  --valid_dataset_name data.val.txt \
  --bbox_level segment \
  --unit_type segment \
  --max_num_units 256 \
  --pretrained_model_path /home/dasom/ROOR/make_weights/microsoft/layoutlmv3-large-2048 \
  --save_model_dir /home/dasom/ROOR/rop/weights \
  # --batch_size 2 \
  --batch_size 1 \
  # --accumulate_grad_batches 8 \
  --accumulate_grad_batches 2 \
  --max_epochs 500 \
  --learning_rate 2e-5 \
  --log_every_n_steps 1 \
  --keep_checkpoint_max 1 \
  --patience 50 \
  --shuffle true \
  --seed 2024 \
  --gpus 1