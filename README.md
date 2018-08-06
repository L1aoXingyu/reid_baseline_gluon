# ReID_baseline
Baseline model (with bottleneck) for person ReID (using softmax and triplet loss).

We support
- multi-GPU training
- easy dataset preparation
- end-to-end training and evaluation

## Get Started
1. `cd` to folder where you want to download this repo
2. Run `git clone https://github.com/L1aoXingyu/reid_baseline_gluon.git`
3. Install dependencies:
    - [mxnet 1.3.1](http://mxnet.incubator.apache.org/versions/1.2.1/install/index.html)
    ```
    pip install --pre mxnet-cu90
    ```
    - tensorflow (for tensorboard)
    - [MXBoard](https://github.com/awslabs/mxboard)
4. Prepare dataset
    
    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid_baseline
    mkdir data
    ```
    1. Download dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    2. Extract dataset and rename to `market1501`. The data structure would like:
    ```
    market1501/
        bounding_box_test/
        bounding_box_train/
    ```
5. Prepare pretrained model if you don't have
    ```python
    from mxnet import gluon
    gluon.model_zoo.vision.resnet50_v1(pretrained=True)
    ```
    Then it will automatically download model in `~.mxnet/models/`, you should set this path in `config.py`

## Train
You can run 
```bash
bash scripts/train_triplet_softmax.sh
```
in `reid_baseline` folder if you want to train with softmax and triplet loss. You can find others train scripts in `scripts`.

## Results

| loss | rank1 | map |
| --- | --| ---|
| softmax | 87.1% | 67.8% |
| triplet | 88.2% | 73.7% | 
|triplet + softmax | 90.4% | 76.4% |

I find the mxnet.gluon results are a little bit lower than [pytorch](https://github.com/L1aoXingyu/reid_baseline) results, and I cannot get the reason. I would appreciate that if anyone can help me.
