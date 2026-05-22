# Project J: LLM-as-Judge — Meta-Evaluation Study

This repository is a complete, reproducible code-base for an NLP final project on **LLM-as-Judge bias analysis**. It measures how well a small judge model agrees with human preferences and quantifies several common biases:

- position bias,
- length bias,
- style-over-substance bias,
- optional self-preference bias,
- pointwise calibration.

It also implements mitigation strategies:

- position-swap averaging,
- rubric-based judging,
- ensemble judging,
- optional reasoning-style judging prompt.

The repository is designed to run in **Google Colab without a local GPU**. It also includes a deterministic `heuristic` backend so the whole pipeline can be tested instantly on any laptop before running open models.

---

## Folder structure

```text
llm_as_judge_project/
├── configs/default.yaml                 
├── data/judge_benchmark.csv             
├── notebooks/colab_run_project.ipynb    
├── requirements.txt
├── results/                             
├── src/llm_judge/
│   ├── bias.py
│   ├── data.py
│   ├── judges.py
│   ├── metrics.py
│   ├── mitigations.py
│   ├── parsing.py
│   ├── prompts.py
│   ├── run_experiment.py
│   └── visualize.py
└── tests/test_parsing_metrics.py
```

---

## Recommended environment: Google Colab

### Step 1. Upload the zip

Upload the project zip to Google Drive or directly to Colab.

### Step 2. Unzip and install dependencies

In a Colab cell:

```bash
!unzip llm_as_judge_project.zip -d /content/
%cd /content/llm_as_judge_project
!pip install -r requirements.txt
```

### Step 3. Quick test: no GPU needed

```bash
!python -m src.llm_judge.run_experiment --config configs/default.yaml --backend heuristic
```

This produces CSV files and plots in `results/`.

### Step 4. Run an open-source LLM judge

For Colab free GPU, use a small instruction/text-to-text model first:

```bash
!python -m src.llm_judge.run_experiment \
  --config configs/default.yaml \
  --backend hf \
  --model google/flan-t5-base
```

If Colab memory is limited, use:

```bash
!python -m src.llm_judge.run_experiment \
  --config configs/default.yaml \
  --backend hf \
  --model google/flan-t5-small
```
# llm_as_judge_project
