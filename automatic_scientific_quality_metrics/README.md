# Code for the paper "Automatic Evaluation Metrics for Artificially Generated Scientific Research"

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repo contains the code for the paper: ["Automatic Evaluation Metrics for Artificially Generated Scientific Research"]().

![Reviewer Performance](./assets/reviewer_performance.png)

## Dependencies

### Requirements for running the experiments:

* anaconda3/miniconda3

First clone the repo and then create a new conda environment with the necessary requirements:

```
git clone https://github.com/NikeHop/automatic_scientific_quality_metrics.git
cd automatic_scientific_quality_metrics
conda create --name scientific_qm python=3.11
conda activate scientific_qm
pip install -e .
python -c "import nltk; nltk.download('punkt_tab')"
```

All the following commands require to be run in the scientific_qm environment. Results are logged to wandb.

## Datasets


The datasets used in this project can be found on Huggingface:

* [Section Classification](https://huggingface.co/datasets/nhop/academic-section-classification)

* [Scholarly Document Quality Prediction (SDQP)](https://huggingface.co/datasets/nhop/scientific-quality-score-prediction)


This section describes how to obtain the missing parsed pdfs of the submissions, for the datasets:

* openreview-iclr
* openreview-neurips
* openreview-full

You will need tmux. For Ubuntu/Debian install it via:

```
sudo apt update
sudo apt install tmux
```

All commands should be run from the `./automatic_scientific_qm/data_processing` directory. First get [GROBID](https://github.com/kermitt2/grobid) by running:

```
bash ./scripts/setup_grobid.sh
```

Test whether GROBID runs:

```
bash ./scripts/run_grobid.sh
```

If you run into trouble setting up GROBID have a look at the git issues [here](https://github.com/allenai/s2orc-doc2json). If GROBID works, run the script:

```
bash ./scripts/complete_openreview_dataset.sh
```

## Train Section Classifier

To train the section classifier on sections from the ACL-OCL dataset run from the `./automatic_scientific_qm/section_classification directory`:

```
python train.py --config ./configs/train_acl_ocl.yaml
```

Running the code for the first time will embed the dataset using [SPECTER2](https://huggingface.co/allenai/specter2), which takes ~2hr.  

## Train Score Predictors

To train the score prediction models run from the `./automatic_scientific_qm/score_prediction` directory the following command:

```
python train.py --config ./configs/name_of_config.yaml
```

### Citation Count prediction ACL-OCL
For the citation count prediction models on the ACL-OCL dataset use/modify the config `acl_ocl_citation_prediction.yaml`:


| Parameter | Values | 
|-----------|------|
|data/paper_representation | title_abstract, <br> intro, <br> conclusion, <br> background, <br> method, <br> result_experiment, <br> hypothesis |
|data/context_type| no_context, <br> references, <br> full_paper|


### Score prediction OpenReview

For the score prediction models on the OpenReview dataset use/modify the `openreview_score_prediction_*.yaml`:

| Parameter | Values | 
|-----------|------|
|data/dataset | openreview-full, <br> openreview-iclr, <br> openreview-neurips 
|data/score_type | avg_citations_per_month, <br> mean_score, <br> mean_impact  |
|data/paper_representation | title_abstract,<br> hypothesis |
|data/context_type| no_context, <br> references, <br> full_paper|


If the code is run for the first time for each dataset the text of the datasets will be embedded using [SPECTER2](https://huggingface.co/allenai/specter2), which takes ~3hr. 

## Run LLM Reviewers  

All commands should be run from the `./automatic_scientific_qm/llm_reviewing` directory.

**Note 1:** 

This section requires that the missing PDF submissions for `openreview-iclr` and `openreview-neurips` are parsed (see [here](#datasets)). 

**Note 2:** 

This section requires an API key for either OpenAI or Anthropic. If you want to use the Anthropic API, store the API key as an environment variable in `ANTHROPIC_API_KEY`. If you want to use the OpenAI API, store the API key as an environment variable in
`OPENAI_API_KEY`.

**Note 3:** 

Download the necessary data by running

```
bash ./scripts/download.sh
```

### Running Review Models
We run the following two LLM-reviewers on a subset of ICLR-2024 and NeurIPS-2024 submissions:

(1) [Sakana's LLM reviewer](https://arxiv.org/abs/2408.06292) 

The first argument specifies the llm_provider, currently only openai and anthropic are supported

```
bash ./scripts/run_sakana_reviewer.sh openai
```


(2) [Paiwise Comparison Reviewer](https://arxiv.org/abs/2409.04109).

```
bash ./scripts/run_llm_pairwise_reviewer.sh openai
```

(3) Run score prediction models on subsets 

```
bash ./scripts/run_rsp_reviewer.sh
```

## Acknowledgements

The code makes use of the following repos:

* [https://github.com/kermitt2/grobid](https://github.com/kermitt2/grobid) (Apache-2.0 license)
* [https://github.com/allenai/s2orc-doc2json](https://github.com/allenai/s2orc-doc2json) (Apache-2.0 license)
* [https://github.com/SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) (Apache-2.0 license)
* [https://github.com/NoviScl/AI-Researcher](https://github.com/NoviScl/AI-Researcher) (MIT license)

## Citation 

If you make use of this codebase, please cite

```
@article{hopner2025automatic,
  title={Automatic Evaluation Metrics for Artificially Generated Scientific Research},
  author={H{\"o}pner, Niklas and Eshuijs, Leon and Alivanistos, Dimitrios and Zamprogno, Giacomo and Tiddi, Ilaria},
  journal={arXiv preprint arXiv:2503.05712},
  year={2025}
}
```

