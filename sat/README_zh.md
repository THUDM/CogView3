# SAT CogView3 && CogView-3-Plus

本文件夹包含了使用 [SAT](https://github.com/THUDM/SwissArmyTransformer) 权重的推理代码，以及 SAT 权重的微调代码。

该代码是团队训练模型时使用的框架。注释较少，需要认真研究。

## 手把手带你运行模型

### 1. 环境安装

确保你已经正确安装本文件夹中的要求的依赖

```shell
pip install -r requirements.txt
```

### 2. 下载模型权重

以下链接为各个模型权重:

### CogView-3-Plus-3B

+ transformer: https://cloud.tsinghua.edu.cn/d/f913eabd3f3b4e28857c
+ vae: https://cloud.tsinghua.edu.cn/d/af4cc066ce8a4cf2ab79

### CogView-3-Base-3B

+ transformer:
    + cogview3-base: https://cloud.tsinghua.edu.cn/d/242b66daf4424fa99bf0
    + cogview3-base-distill-4step: https://cloud.tsinghua.edu.cn/d/d10032a94db647f5aa0e
    + cogview3-base-distill-8step: https://cloud.tsinghua.edu.cn/d/1598d4fe4ebf4afcb6ae
    + 
  **以上三个版本为替换关系，选择适合自己的版本和对应的配置文件进行运行**

+ vae: https://cloud.tsinghua.edu.cn/d/c8b9497fc5124d71818a/ 

### CogView-3-Base-3B-Relay

+ transformer:
    + cogview3-relay: https://cloud.tsinghua.edu.cn/d/134951acced949c1a9e1/
    + cogview3-relay-distill-2step: https://cloud.tsinghua.edu.cn/d/6a902976fcb94ac48402
    + cogview3-relay-distill-1step: https://cloud.tsinghua.edu.cn/d/4d50ec092c64418f8418/
  
  **以上三个版本为替换关系，选择适合自己的版本和对应的配置文件进行运行**

+ vae: 与 CogView-3-Base-3B 相同

接着，你需要将模型文件排版成如下格式：

```
.cogview3-plus-3b
├── transformer
│   ├── 1
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── imagekl_ch16.pt
```

克隆 T5 模型，该模型不用做训练和微调，但是必须使用。这里，您可以单独下载T5模型，必须是`safetensors`类型，不能是`bin`
类型（否则可能出现错误）。

由于我们在`CogVideoX`中上传过 `safetensors` 格式的T5模型，一个简单的办法是从`CogVideX-2B`模型中克隆模型，然后将其移动到对应的文件夹中。

```shell
git clone https://huggingface.co/THUDM/CogVideoX-2b.git #从huggingface下载模型
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git #从modelscope下载模型
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

通过上述方案，你将会得到一个 safetensor 格式的T5文件，确保在 Deepspeed微调过程中读入的时候不会报错。

```
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files
```

### 3. 修改`configs`中的文件。

这里以`CogView3-Base`为例，提供部分参数的讲解和介绍：

```yaml
args:
  mode: inference
  relay_model: False # 当模型类型为 CogView-3-Relay 时，需要将该参数设置为 True
  load: "cogview3_base/transformer" # 这里填写到transformer文件夹
  batch_size: 8 # 每次推理图像数
  grid_num_columns: 2 # 推理结束后，每个提示词文件夹下会有 grid.png 图片，该数字代表列数。
  input_type: txt # 可以选择命令行输入，或者TXT文件输入
  input_file: configs/test.txt # 如果使用命令行，不需要这个参数
  fp16: True # CogView-3-Plus 模型 需要更换为 bf16 推理
  # bf16: True
  sampling_image_size: 512 # 固定大小，支持512 * 512 分辨率图像
  # CogView-3-Plus 模型可以使用以下两个参数。
  # sampling_image_size_x: 1024 宽 
  # sampling_image_size_y: 1024 高

  output_dir: "outputs/cogview3_base-512x512"
  # # 这个部分是给 CogView-3-Relay 模型使用的，需要将该参数设置为推理模型的输入文件夹，提示词建议与 base 模型生成图片时的提示词的一致。
  # input_dir: "outputs/cogview3_base-512x512" 
  deepspeed_config: { }

model:
  conditioner_config:
  target: sgm.modules.GeneralConditioner
  params:
    emb_models:
      - is_trainable: False
        input_key: txt
        target: sgm.modules.encoders.modules.FrozenT5Embedder
        params:
          model_dir: "google/t5-v1_1-xxl" # T5 safetensors的绝对路径
          max_length: 225 # 支持输入的提示词的最大长度

  first_stage_config:
    target: sgm.models.autoencoder.AutoencodingEngine
    params:
      ckpt_path: "cogview3_base/vae/imagekl_ch16.pt" # VAE PT文件绝对路径
      monitor: val/rec_loss
```

### 4. 推理模型

由于不同的模型需要使用的代码不一样，在这里，我们列出了不同模型的推理代码:

### CogView-3Plus

```shell
  python sample_dit.py --base configs/cogview3_plus.yaml
```

### CogView-3-Base

+ 原始模型

```shell
python sample_unet.py --base configs/cogview3_base.yaml
```

+ 蒸馏版本模型

```bash
python sample_unet.py --base configs/cogview3_base_distill_4step.yaml
```

### CogView-3-Relay

+ 原始模型

```shell
python sample_unet.py --base configs/cogview3_relay.yaml
```

+ 蒸馏版本模型

```shell
python sample_unet.py --base configs/cogview3_relay_distill_1step.yaml 
```

输出图片格式为文件夹，其中，文件夹的名字为生成的序号加提示词的前15个字母，文件夹中包含多张图片，具体数量以 `batch` 参数为准。
其结构应该如下：

```
.
├── 000000000.png
├── 000000001.png
├── 000000002.png
├── 000000003.png
├── 000000004.png
├── 000000005.png
├── 000000006.png
├── 000000007.png
└── grid.png

1 directory, 9 files
```

上述例子中，`batch` 为8。因此，有8张图像并带有一张`grid.png`的图像。