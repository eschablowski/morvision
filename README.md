# MorVision

> Note that this is only tested on Ubuntu and probably does NOT run on windows

> !!!Do **NOT** update models repository here without testing it first!!!

## Installation

### Tensorflow

Look at guides [here](tensorflow.org/install).
> Sources are included in the tf folder and can be installed by following [this](tensorflow.org/install/install_sources) guide.

#### MKL Support

Install from sources and add `--config=mkl` to the bazel build
> Please see [tensorflow.org](https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu) for more information
> MKL can be installed by adding `deb https://apt.repos.intel.com/mkl all main` to `/etc/apt/sources.list/d/intelproducts.list` (Note that `/intelpython`, `/ipp`, `/tbb`, and `/daal` also exist on `apt.repos.intel.com`, although `/intelpython` is down as of April 16th, 2018, see [intel.com](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo))

### Object Detector

1. run
``` bash
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip install matplotlib jupyter
```
2. run
```bash
protoc models/research/object_detection/protos/*.proto --python_out=./models/research
```
3. run or add absolute paths to .bashrc
```bash
cd models/research;export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim;cd ../..
```
4. test by running
```bash
python models/object_detection/builders/model_builder_test.py
```

## Annotation

Just run `npm start` inside of the LabelMeAnnotationTool folder
> This starts a webserver on [localhost:8080](localhost:8080/tool.html)

## Creating Data

```bash
python make_model/make_model.py
```

## Training

### Running Tensorflow

```bash
python models/research/object_detection/train.py --train_dir=./temp --pipeline_config_path=`pwd /make_model/embedded_ssd_mobilenet_v1_coco.config`
```

### Running TensorBoard

```bash
tensorboard --logdir=./temp
```

> Note that to see this open browser to [localhost:6006](localhost:6006)

### Running Evaluation

```bash
python models/research/object_detection/eval.py --eval_dir=./tempEval --pipeline_config_path=/home/elias/Desktop/web/morvision/make_model/embedded_ssd_mobilenet_v1_coco.config
```
