# multimodal-aesthetic-model

This model is built by integrating the usage of LLaVA model for captioning and CLIP model for features extraction. Classification is done by Multilayer Perceptron and human-annotated aesthetic label were used for the ground truth.

To run the training process, run this line on your command.
```
python main.py --scenario multimodal --prompt_type structured
```
The existing scenarios including 'multimodal', 'visual_only' and 'text_only'. For the prompt, existing options are 'contrastive' and 'structured'. The possible combination of the scenarios are shown in this table below.

| Scenario  | Prompt Type |
| ------------- | ------------- |
| Visual Only  | -  |
| Text Only  | Contrastive  |
| Text Only  | Structured  |
| Multimodal  | Contrastive  |
| Multimodal  | Structured  |
