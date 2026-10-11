# BreakfreeInnovator

Describe what you're working on today and get back an achievement that matches it.

A `bert-base-uncased` model is fine-tuned to sort a free-text description of an assignment into one of four categories: writing, math, music, or engineering. The app then replies with an achievement message for that category. When the model is less than 85% confident, it asks you to rephrase.

## Files

| File | Purpose |
|---|---|
| `RawData.JSON` | Example assignment descriptions and achievement responses for each category |
| `TextClassificationDataset.py` | PyTorch `Dataset` that parses the JSON and makes an 80/20 train/test split |
| `model.ipynb` | Fine-tunes BERT and saves `bert_fine_tuned.pt` |
| `achievementGenerator.py` | Interactive command-line app that uses the fine-tuned model |

## Run it

```bash
pip install -r requirements.txt
# run model.ipynb once to produce bert_fine_tuned.pt
python achievementGenerator.py
```
