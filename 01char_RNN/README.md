### 12. RNN基本结构与Char RNN文本生成

**12.5.4 训练模型与生成文字**

训练生成英文的模型：
```
python train.py \
  --input_file data/shakespeare.txt \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

测试模型：
```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \   # 模型的保存路径
  --max_length 1000
```

训练写诗模型：
```
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```


测试模型：
```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

训练生成C代码的模型：
```
python train.py \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

测试模型：
```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path model/linux \
  --max_length 1000
```
模型保存：
```
 --checkpoint_path model/******/    # 模型的保存路径
tensorboard --logdir=模型的保存路径    # tensorboard查看训练过程
```

