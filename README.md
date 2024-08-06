# DeepDFA
Official repository for the paper "DeepDFA: Automata Learning through Neural Probabilistic Relaxations" accepted by the 27th European Conference on Artificial Intelligence (ECAI 2024)

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
