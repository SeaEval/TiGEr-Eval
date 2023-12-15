# TiGEr-Eval: Text Generation Evaluation Toolkit

<p align="center">
  <img src="asset/tiger.png" width="250">
</p>

# Overview

**TiGEr** toolkit for text generation evaluation.


# Installation (from Pypi, recommended)

```
pip install tiger-eval
```

# Installation (from source, unstable version)

```
pip install .
```

# Done

- Cross-lingual Consistency

- Multichoice question evaluation

- Open generation evaluation (with llama-2-7b-chat only)

- BLEU score

# TODO


The toolkit should support various metrics

1. ROUGE, BLEU

2. Model based: BERTScore, BLEURT

3. Open Generation, need to be further improved

4. For multichoice questions. Other matching techniques? E.g. use model to reformat the answer?
