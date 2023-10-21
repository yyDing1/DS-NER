# Introduction

This repository contains the data and code implementation for our paper titled `Towards DS-NER: Unveiling and Addressing Latent Noise in Distant Annotations`.

# Data

All datasets referenced in our paper can be found in the `download` folder, named in the format `{data_source}-{Annotation_method}`.

Data Sources:
1. **CoNLL** is a well-known open-domain NER dataset. It consists of 20,744 sentences collected from 1,393 English news articles and is annotated with four types: PER, ORG, LOC, and MISC.
2. **Webpage** is an NER dataset that contains personal, academic, and computer science conference webpages, covering 783 entities belonging to the four types the same as CoNLL03.
3. **BC5CDR** consists of 1,500 biomedical articles, containing 15,935 Chemical and 12,852 Disease mentions in total.
4. **MIT-Movie** contains 12 different entity types in the movie domain.
5. **MIT-Restaurant** is collected in the restaurant domain and contains 9 entity types.

Annotation Methods:
1. KB-Matching: annotate the dataset with knowledge based and a set hand-craft rules.
2. Dict-Matching: annotate the dataset with dictionaries and a set hand-craft rules.
3. ChatGPT-supervised: utilize ChatGPT, a representative general large language model (LLM), as an annotator.
4. UniNER-supervised: utilize UniNER, a NER-specific large generative language model fine-tuned on massive NER data, as an annotator.
5. Random-Masking: Our studies include synthetic datasets where we mask a certain proportion of entities in well-annotated datasets to verify the effectiveness of our methods.

# Quick start

## train

Execute the training process using the following command:

```bash
bash train.sh conll-ChatGPT [GPU_id] [seed]
```
