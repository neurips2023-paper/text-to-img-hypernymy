## Hypernymy Understanding Evaluation of Text-to-Image Models via WordNet Hierarchy

* `laion/` – code for counting lemmas in laion.
* `notebooks/` – sample jupyter notebooks for analysing the models.
* `scripts/` – code for launching our experiments.
* `src/` – code for our experiments.
* `wordnet_classes/` – files for imagenet class to wnid conversion.

### Scripts

* `generate_images.py` – generates images for synsets from a diffusers text-to-image pipeline.
* `generate_images_glide.py` – generates images for synsets from GLIDE.
* `classify_images.py` – classifies the generated images.
* `calculate_metrics.py` – calculates the metrics from logits.
* `generate_coco.py` – generates MS-COCO samples from a diffusers text-to-image pipeline.
* `generate_coco_glide.py` – generates MS-COCO samples from GLIDE.

### Notebooks

* `human_eval.ipynb` – human evaluation results.
* `metrics.ipynb` – model metrics with std for different classifiers, per-synset comparison between models.

### Suggested use

* Generate synset images from a model using `generate_images.py`.
* Classify them using `classify_images.py`.
* Calculate the metrics using `calculate_metrics.py`. 
  * The metrics proposed in the paper are called `SubtreeInProb` and `SubtreeIS`.

For details on how to run the scripts please read the absl app flags.

Please note that the `num_inference_steps` and `eta` parameters in `generate_images.py` are ignored for [unCLIP](https://huggingface.co/docs/diffusers/api/pipelines/unclip). 
