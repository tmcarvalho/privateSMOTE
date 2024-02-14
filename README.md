## $\epsilon$-PrivateSMOTE

$\epsilon$-PrivateSMOTE is a technique designed for safeguarding against re-identification and linkage attacks, particularly addressing cases with a high re-identification risk. It generates synthetic data via noise-induced interpolation with differential privacy principles to obfuscate high-risk cases.

$\epsilon$-PrivateSMOTE allows having new cases similar to the originals while preserving privacy and maximising predictive utility. Most importanly, $\epsilon$-PrivateSMOTE is a resource efficient and less time-consuming than conventional de-identification approaches such as deep learning and differential privacy-based solutions.


### Instructions
**Input:** Table $T$ with set of $m$ attributes and $n$ instances, number of nearest neighbours $knn$, amount of new cases $N$ (per), amount of noise $\epsilon$, group size $k$ and Quasi-Identifiers (QI) set. 

**Output:** $N * n$ synthetic samples

QI set example: "age ftotinc inctot momloc momrule poprule relateg school"

Example of usage with $\epsilon$-PrivateSMOTE

```python3 code/privatesmote.py --input_file "ds38_train.csv" --knn 1 --per 1 --epsilon 0.5 --k 3 --key_vars age ftotinc inctot momloc momrule poprule relateg school```

Analyse the predicitve performance

``` python3 code/modeling.py --input_file "synth_data/ds38_train_0.1-privateSMOTE_QI0_knn1_per1.csv"```

Analyse the linkability risk

``` python3 code/linkability.py --orig_file "ds38_train.csv" --transf_file "synth_data/ds38_train_0.1-privateSMOTE_QI0_knn1_per1.csv" --control_file "ds38_test.csv" --key_vars age ftotinc inctot momloc momrule poprule relateg school```


### Cite
