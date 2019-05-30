# 津南数字制造算法挑战赛 X-Force 代码

[比赛攻略](https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12281915.0.0.11ba1842na87VQ&postId=57411)

# 推理

```sh
./test_b.sh

./test_c.sh
```

最终结果为 submit/test_b_{timestamp}.json 和  submit/test_c_{timestamp}.json


# 训练模型

默认使用一个GPU训练，如使用多GPU请根据脚本中的提示修改

```sh
./train.sh
```

使用tensorboard查看训练进度 (tensorboard 需要单独安装)
```sh
tensorboard --logdir /data
```


