python3 run_task.py\
    --batch_size 4\
    --gpu_ids 0\
    --lr 1e-5\
    --num_train_epochs 3\
    --warmup_proportion 0.1\
    --weight_decay 0.1\
    --fp16 0\
    --print_step 500\
    --mission train\
    --train_file_name DATA/csqa/train_data.json\
    --dev_file_name DATA/csqa/dev_data.json\
    --test_file_name DATA/csqa/trial_data.json\
    --pred_file_name  /content/drive/MyDrive/CSQA/Models/task_result.json\
    --output_model_dir /content/drive/MyDrive/CSQA/Models/albert-xxlarge\
    --pretrained_model_dir albert-xxlarge-v2\


