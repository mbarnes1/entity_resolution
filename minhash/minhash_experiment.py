import time
import os
import itertools
import math
import argparse
import numpy as np
from hashlib import sha1
from datasketch import MinHash
from random import sample
from sklearn import metrics
####functions

def get_clusters(fn):
    with open(fn,'r') as f:
        f.next()#skip header
        for line in f:
            yield line.split(',')[1:]

def get_lsh(sig):
    for i,band in enumerate(np.array_split(sig,bands)):
        yield sha1("wine" + unicode(band) + "eniw"+unicode(i)).digest()
         
def get_bandwidth(n, t):
        """
        Threshold t = (1/b) ** (1/r) where
        b #bands
        r #rows per band
        n = b * r  #elements in signature
        """
        best = n, 1
        minerr  = float("inf")
        for r in xrange(1, n + 1):
            try:
                b = 1. / (t ** r)
            except:             # Divide by zero, your signature is huge
                return best
            err = abs(n - b * r)
            if err < minerr:
                best = r
                minerr = err
        return best


def connected(seed_ad,c):
    stk=[]
    sigs=doc_to_lsh[seed_ad]
    #get lsh band signatures for this ad and 
    #find match candidates
    edges=0
    cluster=set([seed_ad])
    #get candidates and flatten list
    candidates=set(itertools.chain.from_iterable([lsh_dict[sig] for sig in doc_to_lsh[seed_ad]]))
    #stk=list(candidates)
    stk=candidates
    if len(stk) > 1:#candidates contain more than seed_ad itself
        while len(stk)>0:
            cand=stk.pop()
            if cand in cluster:continue#don't check if we've already seen ad
            m1=hashcorp[cand]
            e=0
            for ad in cluster:
                m2=hashcorp[ad]
                if m2.jaccard(m1) >=threshold:
                    #we could break here if we just needed the connected component. But we need the full number of edges so we'll continue.
                    e+=1
            if e>0:
                edges+=e
                cluster.add(cand)
                candidates=set(itertools.chain.from_iterable([lsh_dict[sig] for sig in doc_to_lsh[cand]]))
                stk.update(candidates)
    #all candidates have been checked, full connected component is resolved. 
    ad2cluster.update({i:c for i in cluster })
    cluster_info[c]=(len(cluster),edges)
    
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run minhash ER experiment with given threshold')
    parser.add_argument("-t", dest="threshold", required=True,type=float,
                                help="threshold for ER", metavar="T")
    parser.add_argument("-s", dest="suffix",
                                help="file suffix", metavar="S")
    parser.add_argument('-match', dest='match', action='store_true')
    parser.add_argument('-noclusters', dest='clusters', action='store_false')
    parser.add_argument("-num", dest="num_lines", required=False,
                                help="number of lines to use", metavar="NUM")
    parser.set_defaults(match=False)
    parser.set_defaults(clusters=True)
    parser.set_defaults(suffix='')
    args = parser.parse_args()

    if not os.path.exists('out'):
        os.makedirs('out')

    calc_match=args.match
    calc_clusters=args.clusters
    suffix=args.suffix

    num_lines=args.num_lines
    num_permutations=100#length of minhash signature
    #choose b and r for lsh such that b*r=num_permutations (roughly)
    threshold=args.threshold

    bandwidth=get_bandwidth(num_permutations, threshold)#r
    bands=int(math.ceil(float(num_permutations)/float(bandwidth)))#b
    print "bands "+str(bands)
    #load cluster info. load text
    print 'load cluster info'
    start_time = time.time()
    fname='/home/scratch/trafficjam/entity_resolution_inputs/rebuild_clusters0_'+num_lines+'.csv'
    line_to_clusters={int(key):int(value) for key,value in get_clusters(fname)}

    print("--- %s seconds ---" % (time.time() - start_time))

    print 'load text'
    start_time = time.time()

    linestoget=frozenset(line_to_clusters.keys())

    fname='/home/scratch/trafficjam/rebuild/processed_phone_stripped_text.txt'
    #no dictionary for corpus necessary. We will just load a list of tuples
    #mycorpus=[(i+1,set(line.split())) for i,line in enumerate(open(fname,'r')) if i+1 in linestoget]
    mycorpus={i+1:set(line.split()) for i,line in enumerate(open(fname,'r')) if i+1 in linestoget}

    print("--- %s seconds ---" % (time.time() - start_time))



    print 'Calculate minhash signatures and lsh signatures'
    start_time = time.time()

    #prepare dictionary of hashes
    hashcorp=dict.fromkeys(linestoget)
    lsh_dict={}
    doc_to_lsh={}
    #compute hashes
    for key,doc in mycorpus.iteritems():
        #compute minhash signature
        m=MinHash(num_perm=num_permutations)
        #for token in doc: m.digest(sha1(token.encode('utf8')))
        for token in doc: m.digest(sha1(token))
        #for token in doc: m.digest(sha1(token.encode('utf8', 'ignore')))
        hashcorp[key]=m
        #compute lsh 
        signatures = [sig for sig in get_lsh(m.hashvalues)]
        #store signatures for this document
        doc_to_lsh[key]=signatures
        #store lsh signature to key
        for sig in get_lsh(m.hashvalues):
            if sig in lsh_dict:
                lsh_dict[sig].append(key)
            else:
                lsh_dict[sig]=[key]

    print("--- %s seconds ---" % (time.time() - start_time))

    #now we'll need to compute connected components
    if calc_clusters:    
        print "compute connected components for threshold: "+str(threshold)
        start_time = time.time()
        ad2cluster={}
        cluster_info={}
        count=0

        #import IPython
        #IPython.embed()
        for i,ad in enumerate(hashcorp):
            #if i%100==0:
            if ad not in ad2cluster:
                connected(ad,count)
                count+=1
        print("--- %s seconds ---" % (time.time() - start_time))
            
        print "write results to file"
        start_time = time.time()
        f=open('out/'+num_lines+'ad2cluster_'+num_lines+'_'+suffix+'.csv','w')
        f.write('ad,cluster\n')
        for key, value in ad2cluster.iteritems():
            f.write(str(key)+','+str(value)+'\n')
        f.close()
        f=open('out/cluster_info_'+num_lines+'_'+suffix+'.csv','w')
        f.write('cluster,size,edges\n')
        for key, value in cluster_info.iteritems():
            f.write(str(key)+','+','.join(map(str,value))+'\n')
        f.close()

        print("--- %s seconds ---" % (time.time() - start_time))

        
    if calc_match:

        #create a balanced, pairwise test set
        
        #first create cluster to ad dictionary
        print "Clac and write pairwise matches"
        clusters_to_lines={}
        for key,value in line_to_clusters.iteritems():
                if value in clusters_to_lines:
                    clusters_to_lines[value].append(key)
                else:
                    clusters_to_lines[value]=[key]

        true_cluster_ids=clusters_to_lines.keys()
        np.random.shuffle(true_cluster_ids)
        num_samples=10000
        if num_samples > len(true_cluster_ids):
            num_samples=len(true_cluster_ids)
        triplets=[]
        #this would be faster as a generator function. But probably not worth the effort.
        for i in xrange(0,num_samples,2):
            clus1=true_cluster_ids[i]
            clus2=true_cluster_ids[i+1]
            l=clusters_to_lines[clus1]
            if len(l)<2:
                continue
            a,b=sample(l,2)
            l=clusters_to_lines[clus2]
            c=np.random.choice(l)
            triplets.append((a,b,c))
        f=open('out/scores_t'+str(threshold)+'_size_'+num_lines+suffix+'.csv','w')
        f.write('jac_error,score,true_label\n')
        results=[]
        for a,b,c in triplets:
            m1,m2,m3=hashcorp[a],hashcorp[b],hashcorp[c]
            #actual jaccard
            s1=mycorpus[a]
            s2=mycorpus[b]
            s3=mycorpus[c]
            jac=float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
            apprx=m1.jaccard(m2)
            results.append(apprx,1)
            f.write(str(abs(jac-apprx))+','+str(apprx)+',1\n')
            jac=float(len(s1.intersection(s3)))/float(len(s1.union(s3)))
            apprx=m1.jaccard(m3)
            results.append((apprx,0))
            f.write(str(abs(jac-apprx))+','+str(apprx)+',0\n')
        f.close()
        import IPython
        IPython.embed()
        scores,label=zip(*results) 
        fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=1)
        print 'AUC_'+str(metrics.auc(fpr, tpr))
        


