# Sequential Modelling of the Evolution of Word Representations for Semantic Change Detection
Python code for the paper *"Sequential Modelling of the Evolution of Word Representations for Semantic Change Detection"* by Tsakalidis & Liakata (EMNLP 2020) [1]. The four folders contain the following:
- **data**: contains the data (words, labels, train/test indices) that were used in the paper;
- **models**: contains the source code for the models that were employed in the experimnents (one file per model);
- **evaluation_scripts**: functions to use for evaluating the performance of a model;
- **synthetic_data_scripts**: used for the synthetic data part of the paper (section 4 of the paper).

## Data
First, download the word vectors from this [link](https://www.dropbox.com/sh/d9cmc8kied74hiv/AABT5z1Z67MJ7KChIXWRUvO9a?dl=0) (~250MB) and extract the *vectors.p* file within the */data/* folder. These vectors have been originally generated in [2] (100-dim represesntations in 14 consecutive years). The rest of the folder contains the following:
- **words.p**: a list with the actual words.
- **labels.p**: list of labels ("static" or "change"); there is one label per word.
- **train_idx**: the indices of the words/vectors/labels that were used for training purposes.
- **test_idx**: the indices of the words/vectors/labels that were used for evaluation purposes.

## Models
This folder contains the code for the models that were tested in [1]. Each file is self-contained (i.e., you can test out a model by calling the respective "main" function). Each model has a **"TEST_ON"** variable which needs to be set. For the definition of the TEST_ON variable and its correct use, refer to the Supplementary Material in [1]. It has been currently set as to operate in the full sequence of word vectors for each model invidivually (i.e., s.t. each model identifies word with altered/shifted semantics during the full time period under consideration).

## Evaluation Scripts
The code for evaluating each model. Use the *"evaluation_NNs.py"* for evaluating the neural-based models; for the rest, use the *"evaluation_baselines.py"*.

### References
[1] Tsakalidis, A. and Liakata, M. 2020. Sequential Modelling of the Evolution of Word Representations for Semantic Change Detection. In Proceedings of EMNLP 2020.

[2] Tsakalidis, A., Bazzi, M., Cucuringu, M., Basile, P. and McGillivray, B., 2019, September. Mining the UK Web Archive for Semantic Change Detection. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019) (pp. 1212-1221).
