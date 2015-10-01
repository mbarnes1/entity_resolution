import time
import os,sys
import itertools
import math
import argparse
import numpy as np
from multiprocessing import Pool
from hashlib import sha1
from datasketch import MinHash
from random import sample
from sklearn import metrics
####functions

def get_clusters(fn):
    with open(fn,'r') as f:
        f.next()#skip header
        for line in f:
            a=line.split(',')
            yield a[0],a[2]

def get_lsh(sig,nbands):
    for i,band in enumerate(np.array_split(sig,nbands)):
        yield sha1("wine" + unicode(band) + "eniw"+unicode(i)).digest()
         
def get_bandwidth(n, tr):
        """
        Threshold tr = (1/b) ** (1/r) where
        b #bands
        r #rows per band
        n = b * r  #elements in signature
        """
        best = n, 1
        minerr  = float("inf")
        for r in xrange(1, n + 1):
            try:
                b = 1. / (tr ** r)
            except:             # Divide by zero, your signature is huge
                return best
            err = abs(n - b * r)
            if err < minerr:
                best = r
                minerr = err
        return best


def connected(seed_ad,lshdict,doc2lsh,t):
    stk=[]
    sigs=doc2lsh[seed_ad]
    #get lsh band signatures for this ad and 
    #find match candidates
    edges=0
    cluster=set([seed_ad])
    #get candidates and flatten list
    candidates=set(itertools.chain.from_iterable([lshdict[sig] for sig in doc2lsh[seed_ad]]))
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
                if m2.jaccard(m1) >=t:
                    #we could break here if we just needed the connected component. But we need the full number of edges so we'll continue.
                    e+=1
            if e>0:
                edges+=e
                cluster.add(cand)
                candidates=set(itertools.chain.from_iterable([lshdict[sig] for sig in doc2lsh[cand]]))
                stk.update(candidates)
    #all candidates have been checked, full connected component is resolved. 
    return cluster,edges 
    
def compute_clusters(obj):
    thr=obj[0]
    bandwidth=get_bandwidth(num_permutations, thr)#r
    bands=int(math.ceil(float(num_permutations)/float(bandwidth)))#b
    print "starting calculations for threshold "+str(thr)+"\nnumber of lsh bands: "+str(bands)
    sys.stdout.flush()

    start_time = time.time()
    doc_to_lsh={}
    lsh_dict={}

    for key,m in hashcorp.iteritems():
        #compute lsh 
        signatures = [sig for sig in get_lsh(m.hashvalues,bands)]
        #store signatures for this document
        doc_to_lsh[key]=signatures
        #store lsh signature to key
        for sig in signatures:
            if sig in lsh_dict:
                lsh_dict[sig].append(key)
            else:
                lsh_dict[sig]=[key]
    print("Calculating lsh signatures for threshold "+str(thr)+" took\n ---%s seconds ---\n" % (time.time() - start_time))
    sys.stdout.flush()

    #now we'll to compute connected components
    start_time = time.time()
    ad2cluster={}
    cluster_info={}
    count=0

    for ad in hashcorp:
        if ad not in ad2cluster:
            cl,ed=connected(ad,lsh_dict,doc_to_lsh,thr)
            ad2cluster.update({i:count for i in cl })
            cluster_info[count]=(len(cl),ed)
            count+=1
    print("Computing connected components for threshold: "+str(thr)+" took\n--- %s seconds ---\n" % (time.time() - start_time))
        
    print "write results to file"
    start_time = time.time()
    f=open('out/ad2cluster_'+num_lines+'_'+str(thr)+'_'+suffix+'.csv','w')
    f.write('ad,cluster\n')
    for key, value in ad2cluster.iteritems():
        f.write(str(key)+','+str(value)+'\n')
    f.close()
    f=open('out/cluster_info_'+num_lines+'_'+str(thr)+'_'+suffix+'.csv','w')
    f.write('cluster,size,edges\n')
    for key, value in cluster_info.iteritems():
        f.write(str(key)+','+','.join(map(str,value))+'\n')
    f.close()
    print("Writing results to files for threshold "+str(thr)+" took:\n--- %s seconds ---\n" % (time.time() - start_time))
    
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run minhash ER experiment with given threshold')
    parser.add_argument("-t", dest="threshold",type=float,
                                help="threshold for ER", metavar="T")
    parser.add_argument("-lt", dest="lt",type=float,
                                help="lower threshold for ER", metavar="TL")
    parser.add_argument("-ut", dest="ut",type=float,
                                help="upper threshold for ER", metavar="TU")
    parser.add_argument("-steps", dest="steps",type=float,
                                help="number of steps between lower and upper threshold", metavar="TSTEP")
    parser.add_argument("-s", dest="suffix",
                                help="file suffix", metavar="S")
    parser.add_argument('-match', dest='match', action='store_true')
    parser.add_argument('-noclusters', dest='clusters', action='store_false')
    parser.add_argument("-numl", dest="num_lines", required=False,
                                help="number of lines to use", metavar="NUML")
    parser.add_argument("-p", dest="nump", required=False,type=int,
                                help="number of processes for multithreading", metavar="NUMP")
    parser.set_defaults(match=False)
    parser.set_defaults(threshold=None)
    parser.set_defaults(lt=None)
    parser.set_defaults(ut=None)
    parser.set_defaults(steps=None)
    parser.set_defaults(nump=1)
    parser.set_defaults(clusters=True)
    parser.set_defaults(suffix='')
    args = parser.parse_args()

    if not os.path.exists('out'):
        os.makedirs('out')

    #set variables 
    num_processes=args.nump

    calc_match=args.match
    calc_clusters=args.clusters
    suffix=args.suffix

    num_lines=args.num_lines
    num_permutations=100#length of minhash signature
    thresholds=[]
    lt=args.lt
    ut=args.ut
    steps=args.steps
    if args.threshold is not None:
        thresholds=[args.threshold]
    else:
        if None in [lt,ut,steps]: 
            print "need lower threshold, upper threshold, and number of steps"
            exit()
        else:
            thresholds=np.linspace(lt, ut, num=steps)

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
    mycorpus=[(i,set(line.lower().split())) for i,line in enumerate(open(fname,'r')) if i in linestoget]
    #make corpus a dictionary. Needed to calculate true jaccard score. 
    #mycorpus={i+1:set(line.lower().split()) for i,line in enumerate(open(fname,'r')) if i+1 in linestoget}

    print("--- %s seconds ---" % (time.time() - start_time))

    print 'Calculate minhash signatures'
    start_time = time.time()

    #prepare dictionary of hashes
    hashcorp=dict.fromkeys(linestoget)
    #compute hashes
    for key,doc in mycorpus:#.iteritems():
        #compute minhash signature
        m=MinHash(num_perm=num_permutations)
        #for token in doc: m.digest(sha1(token.encode('utf8')))
        for token in doc: m.digest(sha1(token))
        #for token in doc: m.digest(sha1(token.encode('utf8', 'ignore')))
        hashcorp[key]=m
    print("--- %s seconds ---" % (time.time() - start_time))

    if calc_clusters:    
        p=Pool(num_processes)
        assignment=[ (x,) for x in thresholds]
        print assignment
        p.map(compute_clusters,assignment)

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
        f=open('out/scores_size_'+num_lines+suffix+'.csv','w')
        f.write('score,true_label\n')
        results=[]
        for a,b,c in triplets:
            m1,m2,m3=hashcorp[a],hashcorp[b],hashcorp[c]
            #actual jaccard
            #s1=mycorpus[a]
            #s2=mycorpus[b]
            #s3=mycorpus[c]
            #jac=float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
            apprx=m1.jaccard(m2)
            results.append((apprx,1))
            #f.write(str(abs(jac-apprx))+','+str(apprx)+',1\n')
            f.write(str(apprx)+',1\n')
            #jac=float(len(s1.intersection(s3)))/float(len(s1.union(s3)))
            apprx=m1.jaccard(m3)
            results.append((apprx,0))
            #f.write(str(abs(jac-apprx))+','+str(apprx)+',0\n')
            f.write(str(apprx)+',0\n')
        f.close()
        #import IPython
        #IPython.embed()
        scores,label=zip(*results) 
        fpr, tpr, thres = metrics.roc_curve(label, scores, pos_label=1)
        print 'AUC_'+str(metrics.auc(fpr, tpr))
        


