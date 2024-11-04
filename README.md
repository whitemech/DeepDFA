# DeepDFA
Official repository for the paper "DeepDFA: Automata Learning through Neural Probabilistic Relaxations" accepted by the 27th European Conference on Artificial Intelligence (ECAI 2024).

## Requirements
- Pythomata
- Pytorch

## How to execute the code
To reproduce the experiments in the paper execute the file `do_experiments.py`.
Run `do_experiments.py --help` for a complete list of the parameters used by the script (below).

```
USAGE: do_experiments.py [flags]
flags:

do_experiments.py:
  --dataset_dir: path to the datasets
    (default: 'datasets_Tomita/')
  --error_rate: portion of errated labels in the training dataset
    (default: '0.0')
    (a number)
  --experiment: which experiment to execute, choose one in the list ['tomita', 'random_dfa']
    (default: 'tomita')
  --ground_truth_dfa_dir: path to save ground truth automatons
    (default: 'DFA_ground_truth/')
  --plot_dir: path to save the plots
    (default: 'Plots/')
  --predicted_dfa_dir: path to save predicted automatons
    (default: 'DFA_predicted_nesy/')
  --results_dir: path to save the results
    (default: 'Results/')
```
Examples:

To learn the Tomita languages from data execute the script with default parameters
```
python do_experiments.py
```
To learn the random DFAs in the directory `random_DFAs` run 
```
 python do_experiments.py --dataset_dir datasets_random_dfa/ --experiment random_dfa
```
To repeat the expermints on a corrupted version of the datasets, with 1% of flipped labels, add to the commands above `--error_rate 0.01`

## Citation
```
@inproceedings{UmiliDeepDFA,
  author       = {Elena Umili and
                  Roberto Capobianco},
  editor       = {Ulle Endriss and
                  Francisco S. Melo and
                  Kerstin Bach and
                  Alberto Jos{\'{e}} Bugar{\'{\i}}n Diz and
                  Jose Maria Alonso{-}Moral and
                  Sen{\'{e}}n Barro and
                  Fredrik Heintz},
  title        = {DeepDFA: Automata Learning through Neural Probabilistic Relaxations},
  booktitle    = {{ECAI} 2024 - 27th European Conference on Artificial Intelligence,
                  19-24 October 2024, Santiago de Compostela, Spain - Including 13th
                  Conference on Prestigious Applications of Intelligent Systems {(PAIS}
                  2024)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {392},
  pages        = {1051--1058},
  publisher    = {{IOS} Press},
  year         = {2024},
  url          = {https://doi.org/10.3233/FAIA240596},
  doi          = {10.3233/FAIA240596},
  timestamp    = {Fri, 25 Oct 2024 12:13:46 +0200},
  biburl       = {https://dblp.org/rec/conf/ecai/UmiliC24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

