# MoFlow: Motion-Guided Flows for Recurrent Rendered Frame Prediction

This is offical repository for our paper, **MoFlow: Motion-Guided Flows for Recurrent Rendered Frame Prediction**<br>
**Authors:** Zhizhen Wu, Zhilong Yuan, Chenyu Zuo, Yazhen Yuan, Yinfan Peng, Guiyang Pu, Rui Wang and Yuchi Huo.<br>
in Transactions on Graphics. <br>
[ACM Library Page](https://dl.acm.org/doi/10.1145/3730400)


<p align="center"><img src=".md_asset/teaser.png"></p>

# Updates & Todos (Keep Updating...)
```
[2025-12-02]: Release code.
[2026-01-11]: Add pretrained models and dataset samples.
[TODO]: Add details on dataset generation.
```
# Setup

1. Make a directory for workspace and clone the repository: 
```
mkdir motion-guided-flow; cd motion-guided-flow
git clone https://github.com/jsnzwu/motion-guided-flow code
cd code
```
2. Install conda env:
```bash
conda update -n base -c defaults conda --yes
conda env create -f env.yaml
```
# Quick Demo

> 
> **Install prerequisite: `huggingface-cli`**  
> ```bash
> curl -LsSf https://hf.co/cli/install.sh | bash
> ```

## 1) Download pretrained models

```bash
./scripts/download_pretrained_models_demo.sh
```

## 2) Download the demo dataset

```bash
./scripts/download_datasets_demo.sh
```
## 3) Run the inference demo

To generate inference results on the **DownTown** scene with the **MoFlow** model:

```bash
python src/test/test_inference.py --mode moflow --config ./config/inference/DT_moflow.yaml --block 2 --video --scene DT_TEST
```

To generate inference results on the **FutureCity** scene with the **MoFlow** model:

```bash
python src/test/test_inference.py --mode moflow --config ./config/inference/FC_moflow.yaml --block 2 --video --scene FC_TEST
```

**`--block`** controls the prediction interval. Setting `--block 2` predicts one frame every other frame, producing an alternating sequence like:
`[rendered, predicted, rendered, predicted, ...]`.


# Dataset Generation

## Export Buffers from UE4
Please refer to [Export Buffers from UE4 in LMV](https://github.com/jsnzwu/learnable-motion-vector?tab=readme-ov-file#export-buffers-from-ue4) for details.

## Preprocess
Please refer to [Compress Raw Files into NPZ Files in LMV](https://github.com/jsnzwu/learnable-motion-vector?tab=readme-ov-file#compress-raw-files-into-npz-files) for details.
### Run the script
```
python src/test/test_export_buffer.py --config config/export/export_st.yaml
```
## Manage the dataset paths
The config is at `/config/includes/dataset_export_job.yaml`

The dataset path `/path/to/npz` should be accepted with full path: (eg. `/path/to/npz/FC_T/FC_05_720/<BufferName>/<Index>.npz`)
## Training
### Training from Scratch
Run the script:
```
python src/test/test_train.py --config config/entry/single_moflow.yaml --train [--test]
```
- `--train`: training stage only
- `--train --test`: training and testing
### Resume the Previous Training
Requirements: generated training result in the following directory structure, e.g.
```
job_name (e.g., single_moflow)/
|-- time_stamp(e.g., 2025-MM-DD_HH-MM-SS)
| |-- writer (logs in tensorboard format)
| |-- log (logs in text format)
| |-- model (the models' pt of the best and the newest)
| |-- checkpoint (the last checkpoint)
| |-- history_models (all history models)
| |-- history_checkpoints (all history checkpoints)
```
Run the script with `--train --resume`:
```
 python src/test/test_train.py --config config/entry/single_moflow.yaml --train --resume
```
## Testing
Requirements: generated training result.
run the script with `--test`:
```
python src/test/test_train.py --config config/entry/single_moflow.yaml --train [--test]
```
## Inference
Run single-frame prediction:

`python src/test/test_inference.py --mode moflow --block 2 --scene <SceneName> [--image | --video]`


Run multi-frame prediction (3×):

`python src/test/test_inference.py --mode moflow --block 4 --scene <SceneName> [--image | --video]`

Arguments

- `--block <int>`
Total number of frames in one block, including the rendered frame and all extrapolated frames (`block_size = rendred + N  x extrapolated_frames`).

- `--scene <str>`
Name of the sequence to evaluate (e.g., FC_05).

- `--image / --video` (Output format, choose exactly one):
    - `--image:` export prediction frames as images for a selected time interval.
    - `--video`: export prediction frames as images for the entire sequence.



# BibTeX
If you find our paper helpful, please consider citing:
```
@article{wu25motion,
  author = {Wu, Zhizhen and Yuan, Zhilong and Zuo, Chenyu and Yuan, Yazhen and Peng, Yifan and Pu, Guiyang and Wang, Rui and Huo, Yuchi},
  title = {MoFlow: Motion-Guided Flows for Recurrent Rendered Frame Prediction},
  year = {2025},
  issue_date = {April 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {44},
  number = {2},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3730400},
  doi = {10.1145/3730400},
  journal = {ACM Trans. Graph.},
  month = apr,
  articleno = {22},
  numpages = {18},
  keywords = {Real-time rendering, frame extrapolation, spatial-temporal}
}
```

# Contact
`:)` If you have any questions or suggestions about this repo, please feel free to contact me (jsnwu99@gmail.com).
