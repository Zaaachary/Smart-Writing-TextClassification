python CODE\run_task.py^
    --task_name BERT_MLP^
    --mission test^
    --fp16 0^
    --gpu_ids 0^
    --print_step 100^
    ^
    --lr 1e-5^
    --batch_size 8^
    --num_train_epochs 3^
    --warmup_proportion 0.1^
    --weight_decay 0.1^
    ^
    --dataset_dir DATA/^
    --pred_file_dir  DATA/result/task_result.json^
    --model_save_dir DATA/result/TCmodel/^
    --PTM_model_vocab_dir D:\CODE\Python\Transformers-Models\bert-base-chinese^