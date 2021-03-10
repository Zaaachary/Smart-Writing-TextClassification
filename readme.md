# CSQA


## Model Architecture

BERT + MLP


## Usage

### package requirement

1. python 第三方库
tqdm
transformers 4.2.2
pytorch 1.7.0  请根据 GPU 和 CUDA 下载对应版本
tqdm
sklearn

2. 预训练模型下载
https://huggingface.co/bert-base-chinese/tree/main
下载 config.json, pytorch_model.bin, tokenizer.json, tokenizer_config.son, vocab.txt 到 bert-base-chinese 文件夹里

### run

修改 shell 中的 .bat 文件， 主要修改PTM_model_vocab_dir 和 model_save_dir

在 readme.md 相同层级的目录调用 Shell 里的 .bat 文件

训练: run_train.bat   (train + dev)
测试：run_test.bat  (test)
未完成 ↓
预测：run_predict (target)
