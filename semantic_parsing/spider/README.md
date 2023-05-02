```bash
python -m language.nqg.tasks.spider.generate_gold \
  --input ../compositional-generalization/data/spider-ssp-schema-tuning/dev.csv \
  --output ../compositional-generalization/data/spider-ssp-schema-tuning/dev-gold.txt
```

```bash
python evaluation.py \
  --gold ../../data/spider-ssp-schema-tuning/dev-gold.txt \
  --pred ../../outputs/spider-ssp-schema-tuning-bart-large/out-220117_212605_47fd32df/generated_predictions.txt \
  --etype all --db ../../data/spider/database/ \
  --table ../../data/spider/tables.json 2>&1 | tee ../../outputs/spider-ssp-schema-tuning-bart-large/out-220117_212605_47fd32df/generated_predictions_eval_output.txt
```