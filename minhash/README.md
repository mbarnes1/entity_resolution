# MinHash experiments
Use Minhash to estimate jaccard similarity between documents in order to find near duplicates. Then calculate connected components 

## Dependencies
This script uses https://github.com/ekzhu/datasketch  for some computations

## How to
    $python2.7 minhash_experiment.py -lt 0.5 -ut 1.0 -steps 30 -num 1000000 -p 15
