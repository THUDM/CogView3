# SAT CogView3 & CogView-3-Plus

[Read this in Chinese](./README_zh.md)

This folder contains the inference code using the [SAT](https://github.com/THUDM/SwissArmyTransformer) weights, as well as fine-tuning code for SAT weights.

The code is the framework used by the team during model training. There are few comments, so it requires careful study.

## Step-by-step guide to running the model

### 1. Environment setup

Ensure you have installed the dependencies required by this folder:

```shell
pip install -r requirements.txt
```

### 2. Download model weights

The following links are for different model weights:

### CogView-3-Plus-3B

+ transformer: https://cloud.tsinghua.edu.cn/d/f913eabd3f3b4e28857c
+ vae: https://cloud.tsinghua.edu.cn/d/af4cc066ce8a4cf2ab79

### CogView-3-Base-3B

+ transformer:
    + cogview3-base: https://cloud.tsinghua.edu.cn/d/242b66daf4424fa99bf0
    + cogview3-base-distill-4step: https://cloud.tsinghua.edu.cn/d/d10032a94db647f5aa0e
    + cogview3-base-distill-8step: https://cloud.tsinghua.edu.cn/d/1598d4fe4ebf4afcb6ae
  
  **These three versions are interchangeable. Choose the one that suits your needs and run it with the corresponding configuration file.**

+ vae: https://cloud.tsinghua.edu.cn/d/c8b9497fc5124d71818a/ 

### CogView-3-Base-3B-Relay

+ transformer:
    + cogview3-relay: https://cloud.tsinghua.edu.cn/d/134951acced949c1a9e1/
    + cogview3-relay-distill-2step: https://cloud.tsinghua.edu.cn/d/6a902976fcb94ac48402
    + cogview3-relay-distill-1step: https://cloud.tsinghua.edu.cn/d/4d50ec092c64418f8418/
  
  **These three versions are interchangeable. Choose the one that suits your needs and run it with the corresponding configuration file.**

+ vae: Same as CogView-3-Base-3B

Next, arrange the model files into the following format:

```
.cogview3-plus-3b
├── transformer
│   ├── 1
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── imagekl_ch16.pt
```

Clone the T5 model. This model is not used for training or fine-tuning but is necessary. You can download the T5 model separately, but it must be in `safetensors` format, not `bin` format (otherwise an error may occur).

Since we have uploaded the T5 model in `safetensors` format in `CogVideoX`, a simple way is to clone the model from the `CogVideoX-2B` model and move it to the corresponding folder.

```shell
git clone https://huggingface.co/THUDM/CogVideoX-2b.git
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

With this setup, you will have a safetensor format T5 file, ensuring no errors during Deepspeed fine-tuning.

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

### 3. Modify the files in `configs`.

Here is an example using `CogView3-Base`, with explanations for some of the parameters:

```yaml
args:
  mode: inference
  relay_model: False # Set to True when using CogView-3-Relay
  load: "cogview3_base/transformer" # Path to the transformer folder
  batch_size: 8 # Number of images per inference
  grid_num_columns: 2 # Number of columns in grid.png output
  input_type: txt # Input can be from command line or TXT file
  input_file: configs/test.txt # Not needed for command line input
  fp16: True # Set to bf16 for CogView-3-Plus inference
  # bf16: True
  sampling_image_size: 512 # Fixed size, supports 512x512 resolution images
  # For CogView-3-Plus, use the following:
  # sampling_image_size_x: 1024 (width)
  # sampling_image_size_y: 1024 (height)

  output_dir: "outputs/cogview3_base-512x512"
  # This section is for CogView-3-Relay. Set the input_dir to the folder with base model generated images.
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
          model_dir: "google/t5-v1_1-xxl" # Path to T5 safetensors
          max_length: 225 # Maximum prompt length

  first_stage_config:
    target: sgm.models.autoencoder.AutoencodingEngine
    params:
      ckpt_path: "cogview3_base/vae/imagekl_ch16.pt" # Path to VAE PT file
      monitor: val/rec_loss
```

### 4. Running the model

Different models require different code for inference. Here are the inference commands for each model:

### CogView-3Plus

```shell
python sample_dit.py --base configs/cogview3_plus.yaml
```

### CogView-3-Base

+ Original model

```shell
python sample_unet.py --base configs/cogview3_base.yaml
```

+ Distilled model

```bash
python sample_unet.py --base configs/cogview3_base_distill_4step.yaml
```

### CogView-3-Relay

+ Original model

```shell
python sample_unet.py --base configs/cogview3_relay.yaml
```

+ Distilled model

```shell
python sample_unet.py --base configs/cogview3_relay_distill_1step.yaml 
```

The output image format will be a folder. The folder name will consist of the sequence number and the first 15 characters of the prompt, containing multiple images. The number of images is based on the `batch` parameter. The structure should look like this:

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

In this example, the `batch` size is 8, so there are 8 images along with one `grid.png`.
