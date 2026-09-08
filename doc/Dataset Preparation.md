# aisbench测试指导

#### 安装aisbench

aisbench开源仓：

[GitHub - AISBench/benchmark: AISBench Benchmark is a model evaluation tool built on OpenCompass, compatible with OpenCompass’s configuration system, dataset structure, and model backend implementation, while extending support for service-based models. · GitHub](https://github.com/AISBench/benchmark)

根据开源仓主页的“工具安装”章节完成aisbench安装

#### 下载aisbench测试工具：

[GitHub - rayn-zzz/aisbench_auto_tools_prefix · GitHub](https://github.com/rayn-zzz/aisbench_auto_tools_prefix/tree/main)

按照“使用方法”章节，修改config.py脚本：

关键修改：WORK_PATH 、MODEL_NAME 、MODEL_PATH 、HOST_IP 、HOST_PORT 

如需开启prefixcache功能，必须修改POD_INFO 

#### 常用指令

默认使用开源gsm8k数据集，构造成所需输入长度。如果不需要开启prefixcache功能，执行指令1。开启prefixcache后需要构造前缀重复数据集，执行指令2.

1、测试2k/2k不带前缀的gsm8k数据集性能

```shell
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10
```

2、测试2k/2k带前缀的gsm8k数据集性能，前缀个数1，数据集前缀重复率50%，dp 2，**先预热前缀**

```shell
python3 aisbench_test.py --input_len 2048 --output_len 2048 --data_num 160 --concurrency 40 --request_rate 10 --dataset_type prefix_cache --repeat_rate 0.5 --prefix_test --dp 2
```

#### 实测指令

##### GLM5.2模型 

 Case1：128k+1k prefix cache 90%   （dp数表示pd分离模型下，p节点的dp域数量，如果不用pd分离部署，则忽略）

```shell
python3 aisbench_test.py --input_len 135000 --output_len 1024 --data_num xxx --concurrency xx --request_rate xx --dataset_type prefix_cache --repeat_rate 0.9 --prefix_test --dp x
```

  Case2:  16k+1k  prefix cache 0%     

```shell
python3 aisbench_test.py --input_len 16384 --output_len 1024 --data_num xxx --concurrency xx --request_rate xx
```

##### deepseekv4-flash模型

16K+1K prefix cache 0%

```shell
python3 aisbench_test.py --input_len 16384 --output_len 1024 --data_num xxx --concurrency xx --request_rate xx
```

