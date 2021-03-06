import numpy as np
import time
import random
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
import json
from multiprocessing import Process,Manager
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import time

from itertools import permutations
from itertools import combinations

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

tf.enable_v2_behavior()

from joblib import Parallel, delayed

ds_train, ds_info = tfds.load(
    'forest_fires',
    split=['train'],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)

ds_numpy = tfds.as_numpy(ds_train)
profile_features = []
labels = []
for ex in ds_numpy[0]:
  profile_features.append([ex[1][entry] for entry in ex[1]])
  labels.append(ex[0])

print("dataset size:",len(labels))

"""## Limited Data Experiments"""
print("begin experiment")
num_trials = 10
this_train_sizes = np.linspace(1/len(labels),1,len(labels))
results = [0 for i in range(len(this_train_sizes)*num_trials)]
results = Manager().list([0 for i in range(len(this_train_sizes)*num_trials)])
val_results = Manager().list([0 for i in range(len(this_train_sizes)*num_trials)])


def gen_eval_set(profile_features,labels,eval_set):
  eval_features = np.array([profile_features[idx] for idx in eval_set])
  eval_labels = np.array([labels[idx] for idx in eval_set])
  return eval_features,eval_labels,eval_set

def D_func(X):
  return np.linalg.det(np.dot(X.T,X))

def dopt_eval_set(num_trials,size,profile_features,labels):
  dopt_val_max = float('-inf')
  best_set = None
  for j in range(num_trials):
    cur_set = set(random.sample([i for i in range(len(labels))],int(size*len(labels))))
    cur_features,cur_labels,_ = gen_eval_set(profile_features,labels,cur_set)
    tmp_val = D_func(cur_features)
    if tmp_val > dopt_val_max:
      dopt_val_max = tmp_val
      best_set = cur_set
  return best_set


def get_costs(cur_reg,samples,sample_costs,eval_features,eval_labels,profile_features,labels,cur_X_train,cur_y_train,num_to_profile,available_list,start_index,num_work,K):
  for j in range(start_index,start_index+num_work):
    new_sample_idx = samples[j][2]
    new_X_train = np.array([profile_features[idx] for idx in new_sample_idx])
    new_y_train = np.array([labels[idx] for idx in new_sample_idx])
    reg = RandomForestRegressor(n_estimators=100).fit(np.concatenate((cur_X_train,new_X_train)),np.concatenate((cur_y_train,cur_reg.estimators_[K].predict(new_X_train))))
    cost = mean_absolute_error(eval_labels,reg.predict(eval_features))
    sample_costs[j] = cost
     
def run_trial(profile_features,labels,this_train_sizes,results,val_results,n):
  print("trial",n)
  random.seed(n)
  np.random.seed(n)
  profile_features,labels = shuffle(profile_features,labels)
  cur_X_train,cur_y_train = profile_features[0:int(np.ceil(len(labels)*this_train_sizes[0]))],labels[0:int(np.ceil(len(labels)*this_train_sizes[0]))]
  available_sample = set([i for i in range(int(np.ceil(len(labels)*this_train_sizes[0])),len(labels))])
  cur_reg = RandomForestRegressor(n_estimators=100).fit(cur_X_train,cur_y_train)
  results[n*len(this_train_sizes)] += mean_absolute_error(labels,cur_reg.predict(profile_features))
  eval_set = dopt_eval_set(1000000,0.3,profile_features,labels)
  eval_features,eval_labels,eval_set = gen_eval_set(profile_features,labels,eval_set)
  val_results[n*len(this_train_sizes)] += mean_absolute_error(eval_labels,cur_reg.predict(eval_features))
  for i in range(1,len(this_train_sizes)):
    start_t = time.time()
    num_to_profile = max(1,int(np.floor(len(labels)*(this_train_sizes[i]-this_train_sizes[i-1]))))
    available_list = shuffle(list(available_sample)) #for num_to_profile = 1
    eval_features,eval_labels,eval_set = gen_eval_set(profile_features,labels,eval_set)
    K = random.randint(0,99)
    samples = []
    for j in range(len(available_sample)):
      if num_to_profile == 1:
        new_sample_idx = [available_list[j]]
      else:
        new_sample_idx = random.sample(available_sample,min(num_to_profile,len(available_sample)))
      new_X_train = np.array([profile_features[idx] for idx in new_sample_idx])
      new_y_train = np.array([labels[idx] for idx in new_sample_idx])
      samples.append((new_X_train,new_y_train,new_sample_idx)) 
    sample_costs = Manager().list([0 for j in range(len(available_list))])
    sub_procs = []
    total_work = len(available_list)
    num_cpus = 32
    num_work = int(np.ceil(total_work/num_cpus))
    for j in range(num_cpus):
      p = Process(target=get_costs,args=(cur_reg,samples,sample_costs,eval_features,eval_labels,profile_features,labels,cur_X_train,cur_y_train,num_to_profile,available_list,j*num_work,min(num_work,total_work),K))
      p.start()
      sub_procs.append(p)
      total_work = max(0,total_work - num_work)
    for j in range(num_cpus):
      sub_procs[j].join()

    best_sample_idx = np.argmin(sample_costs)
    available_sample = available_sample - set(samples[best_sample_idx][2])
    cur_X_train = np.concatenate((cur_X_train,samples[best_sample_idx][0]))
    cur_y_train = np.concatenate((cur_y_train,samples[best_sample_idx][1]))
    cur_reg = RandomForestRegressor(n_estimators=100).fit(cur_X_train,cur_y_train)
    results[n*len(this_train_sizes) + i] += mean_absolute_error(labels,cur_reg.predict(profile_features))
    val_results[n*len(this_train_sizes) + i] += mean_absolute_error(eval_labels,cur_reg.predict(eval_features))
    print(time.time() - start_t)

procs = []
for n in range(num_trials):
  p = Process(target=run_trial, args=(profile_features,labels,this_train_sizes,results,val_results,n))
  p.start()
  procs.append(p)
for n in range(num_trials):
  procs[n].join()

results = np.array(results).reshape((num_trials,len(this_train_sizes)))
val_results = np.array(val_results).reshape((num_trials,len(this_train_sizes)))

avg_results = np.sum(results,axis=0) / num_trials
min_results = np.min(results,axis=0)
max_results = np.max(results,axis=0)

avg_val_results = np.sum(val_results,axis=0) / num_trials
min_val_results = np.min(val_results,axis=0)
max_val_results = np.max(val_results,axis=0)

json.dump(avg_results.tolist(),open("avg_doptdatalimit30_boostrapped_predmae_10sim.json","w"))
json.dump(min_results.tolist(),open("min_doptdatalimit30_boostrapped_predmae_10sim.json","w"))
json.dump(max_results.tolist(),open("max_doptdatalimit30_boostrapped_predmae_10sim.json","w"))

json.dump(avg_val_results.tolist(),open("avg_val_doptdatalimit30_boostrapped_predmae_10sim.json","w"))
json.dump(min_val_results.tolist(),open("min_val_doptdatalimit30_boostrapped_predmae_10sim.json","w"))
json.dump(max_val_results.tolist(),open("max_val_doptdatalimit30_boostrapped_predmae_10sim.json","w"))


json.dump(this_train_sizes.tolist(),open("trainsize_doptdatalimit30_bootstrapped_predmae_10sim.json","w"))
