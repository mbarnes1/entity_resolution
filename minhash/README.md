# MinHash experiments
Use Minhash to estimate jaccard similarity between documents in order to find near duplicates. Then calculate connected components 

## Dependencies
This script uses https://github.com/ekzhu/datasketch  for some computations

## How to
    $python2.7 minhash_experiment.py -t 0.8 -match -num 1000000
    
