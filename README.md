# Benchmarking Long-tail Generalization with Likelihood Splits (Findings of EACL 2023)

This repository contains code for the paper:

> Benchmarking Long-tail Generalization with Likelihood Splits.  
> Ameya Godbole and Robin Jia  
> Findings of the Association for Computational Linguistics: EACL 2023.

## TODO

- [ ] Push main code components
- [ ] Add run scripts for main experiments
- [ ] Update instructions in README

## Citation

```
@inproceedings{godbole-jia-2023-benchmarking,
    title = "Benchmarking Long-tail Generalization with Likelihood Splits",
    author = "Godbole, Ameya  and
      Jia, Robin",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.71",
    pages = "933--953",
    abstract = "In order to reliably process natural language, NLP systems must generalize to the long tail of rare utterances. We propose a method to create challenging benchmarks that require generalizing to the tail of the distribution by re-splitting existing datasets. We create {`}Likelihood Splits{'} where examples that are assigned lower likelihood by a pre-trained language model (LM) are placed in the test set, and more likely examples are in the training set. This simple approach can be customized to construct meaningful train-test splits for a wide range of tasks. Likelihood Splits surface more challenges than random splits: relative error rates of state-of-the-art models increase by 59{\%} for semantic parsing on Spider, 93{\%} for natural language inference on SNLI, and 33{\%} for yes/no question answering on BoolQ, on our splits compared with the corresponding random splits. Moreover, Likelihood Splits create fairer benchmarks than adversarial filtering; when the LM used to create the splits is also employed as the task model, our splits do not unfairly penalize the LM.",
}
```