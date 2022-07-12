# BOW-BERT

Our BOW-BERT model is hosted on the huggingface model hub:



```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load model
model = AutoModelForSequenceClassification.from_pretrained('dmrau/bow-bert')
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# tokenize query and passage and concatenate them
inp = tokenizer(['this is a query','query a is this'], ['this is a passage', 'passage a is this'], return_tensors='pt')
# get estimated score
print('score', model(**inp).logits[:, 1])

### outputs identical scores for different 
### word orders as the model is order invariant:
# scores: [-2.9463, -2.9463]
```
