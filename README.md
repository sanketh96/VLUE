# VLUE: A Multi-task Multi-Dimension Benchmark for Evaluating Vision-Language Pre-training

[**Tasks**](#tasks-and-languages) | [**Download**](#download-the-data) |
[**Baselines**](#build-a-baseline-system) |
[**Leaderboard**](https://vlue-benchmark.github.io/VLUE-website/leaderboard.html) |
[**Website**](https://vlue-benchmark.github.io/VLUE-website) |
[**Paper**](https://arxiv.org/pdf/2003.11080.pdf) 

This repository contains information about the Vision-Language Understanding Evaluation (VLUE) benchmark, instructions for downloading data, and
implementations of baseline systems for the benchmark.

# Introduction

The Vision-Language Understanding Evaluation (VLUE) benchmark is a benchmark for the evaluation of the generalization ability and efficiency-performance trade-off of pre-trained vision-language models. It benchmarks the performance of popular vision-language pre-trained models on the VLUE tasks and also provides a suite of newly croudsourced out-of-distribution (OOD) test sets for each of the four tasks to benchmark the true generalization ability of vision-language pre-trained models.

For a full description of the benchmark, see [the paper](https://arxiv.org/abs/2003.11080).

# Tasks and OOD Test Sets

VLUE covers 4 representative vision-and-language understanding tasks including Image-Text Retrieval, Visual Grouding, Visual Reasoning, and Visual Question Answering. The full description of tasks and datasets can be seen in the image below.  

![The datasets used in XTREME](vlue_tasks.png)

We crowdsource OOD test sets for the 4 tasks in the VLUE benchmark using the raw images from the [MaRVL dataset](https://marvl-challenge.github.io), which contains images from a diverse distribution across countries and cultures. We hire human annotators to contruct test examples for the four tasks using the raw images following the annotation instructions from the original datasets.

# Download the data

We provide intructions to download the data for both the original in-domain datasets and the private OOD test sets.

The images and annotations for the train, dev, and test set of the in-domain datasets can be downloaded [here](https://drive.google.com/file/d/1XFz1Vtz7MCBLn4_1QEojhFJ5Iw3eH3X4/view?usp=sharing).


# Build a baseline system

We provide the examples to run VLUE test on [X-VLM](https://github.com/zengyan-97/X-VLM) as follows: 
```angular2html
cp -r data/ xvlm/data/vlue_released
cd xvlm

python3 run.py --task "eval_vlue_itr" --dist "1" --evaluate  --output_dir "output/" --checkpoint "itr_coco/checkpoint_9.pth"

python3 run.py --task "eval_vlue_vqa" --dist "1" --evaluate  --output_dir "output/" --checkpoint "vqa/model_state_epoch_9.th"

python3 run.py --task "eval_vlue_nlvr" --dist "1" --evaluate  --output_dir "output/" --checkpoint "nlvr/nlvr_ft/checkpoint_best.pth"

python3 run.py --task "eval_vlue_refcoco" --dist "1" --evaluate  --output_dir "output/" --checkpoint "refcoco_bbox/checkpoint_best.pth"

python3 run.py --task "eval_vlue_refcoco_weakly" --dist "1" --evaluate  --output_dir "output/" --checkpoint "refcoco/checkpoint_best.pth"
```



# Leaderboard Submission

## Submissions
To submit your predicitons to [**VLUE**](https://vlue-benchmark.github.io/VLUE-website) following the instructions in the [**VLUE Learderboard**](https://vlue-benchmark.github.io/VLUE-website/leaderboard.html).


# Paper

If you use our benchmark or the code in this repo, please cite our paper `\cite{hu2020xtreme}`.
```
@article{xx,
      author    = {xx},
      title     = {xx},
      journal   = {CoRR},
      volume    = {abs/2003.11080},
      year      = {2020},
      archivePrefix = {arXiv},
      eprint    = {2003.11080}
}
```
Please consider including a note similar to the one below to make sure to cite all the individual datasets in your paper.

We experiment on the VLUE benchmark `\cite{xx}`, a multi-task multi-dimension benchmark for visual-language understanding evaluation consisting of data from the MSCOCO Caption `\cite{chen2015microsoft}`, RefCOCO `\cite{yu2016modeling}`, NLVR2 `\cite{suhr2018corpus}`, VQA 2.0 `\cite{goyal2017making}` datasets, and raw images from the MaRVL dataset `\cite{goyal2017making}` for the private OOD test set.  We provide their BibTex information as follows.
```
@article{chen2015microsoft,
  title={Microsoft coco captions: Data collection and evaluation server},
  author={Chen, Xinlei and Fang, Hao and Lin, Tsung-Yi and Vedantam, Ramakrishna and Gupta, Saurabh and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
  journal={arXiv preprint arXiv:1504.00325},
  year={2015}
}

@inproceedings{yu2016modeling,
  title={Modeling context in referring expressions},
  author={Yu, Licheng and Poirson, Patrick and Yang, Shan and Berg, Alexander C and Berg, Tamara L},
  booktitle={European Conference on Computer Vision},
  pages={69--85},
  year={2016},
  organization={Springer}
}

@article{suhr2018corpus,
  title={A corpus for reasoning about natural language grounded in photographs},
  author={Suhr, Alane and Zhou, Stephanie and Zhang, Ally and Zhang, Iris and Bai, Huajun and Artzi, Yoav},
  journal={arXiv preprint arXiv:1811.00491},
  year={2018}
}

@inproceedings{goyal2017making,
  title={Making the v in vqa matter: Elevating the role of image understanding in visual question answering},
  author={Goyal, Yash and Khot, Tejas and Summers-Stay, Douglas and Batra, Dhruv and Parikh, Devi},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={6904--6913},
  year={2017}
}

@article{liu2021visually,
  title={Visually Grounded Reasoning across Languages and Cultures},
  author={Liu, Fangyu and Bugliarello, Emanuele and Ponti, Edoardo Maria and Reddy, Siva and Collier, Nigel and Elliott, Desmond},
  journal={arXiv preprint arXiv:2109.13238},
  year={2021}
}
```
