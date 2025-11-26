dataset usage:
```python
    # currently only test_mode and train split supported
    dataloader = HowToChangeDataLoader(split='train', test_mode=True)

    for data in dataloader:
        data["frames"] #Tensor(T, C, H, W), video frames in 224*224 resolution
        data["labels"] #Tensor(T), labels[i] == 1 if frames[i] is in end_state
        data["fps"]    #float
        data["osc"]    #str,  ocs in format verb_noun
        data["verb"]   #str
        data["noun"]   #str
```

Ideas for model implementation/training
- running a model directly from raw video frames sounds strange to me... maybe only run model every x frames and extract features from the past x frames using videoMAE https://huggingface.co/docs/transformers/en/model_doc/videomae#transformers.VideoMAEModel then use extracted features as model input
- remember text is also an input, need to choose a text encoder as well. 
- (future extension) we could improve the model with some differentiable text input. for example, for a state change *melting butter*, ask the LLM ahead of time "what does *melting butter* looks like when it's not completed or in progress" and "what does *melting butter* looks like when it's done". LLM might generate something like "the butter looks solid" for when it's in progress, and "butter is completely liquid and sizzling" for when it's done. Save those descriptions in a bank ahead of time. Then during training/inference, our model decides which of those descriptions the visual features are closer to. 

Dataset considertaions (no need to fix):
- there is class imbalance (a lot more 0s than 1s), which may make the model prone to predicting 0 (?)
- there are also irrelevant frames.


