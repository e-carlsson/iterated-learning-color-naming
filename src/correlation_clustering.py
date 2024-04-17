
import numpy as np
import random 
import sys

def max_correlation(my_graph, my_K, my_itr_num):

    my_N = np.size(my_graph,0)

    best_Obj = -sys.float_info.max 
    best_Sol = np.zeros((my_N,), dtype = int)

    for itr in range(0,my_itr_num):
  
        cur_Sol = np.zeros((my_N,), dtype = int) 
        
        for i in range(0,my_N):
            cur_Sol[i] = random.randint(0,my_K-1)

        # calc total cost
        cur_Obj = 0.0
        for i in range(0,my_N):
            for j in range(0,i):
                if cur_Sol[i] == cur_Sol[j]:
                    cur_Obj = cur_Obj + my_graph[i,j]


        old_Obj = cur_Obj - 1

        while cur_Obj-old_Obj > sys.float_info.epsilon:
            old_Obj = cur_Obj

            order = list(range(0,my_N))
            random.shuffle(order)
            
            for i in range(0,my_N):
                cur_Ind = order[i]
                temp_Objs = np.zeros((my_K,), dtype = float)
                
                for j in range(0,my_N): 
                    if j != cur_Ind:
                        temp_Objs[cur_Sol[j]] = temp_Objs[cur_Sol[j]] + my_graph[cur_Ind,j]


                sep_Obj = temp_Objs[cur_Sol[cur_Ind]]
                temp_Objs[cur_Sol[cur_Ind]] = cur_Obj
                
                for k in range(0,my_K):
                    if k != cur_Sol[cur_Ind]:
                        temp_Objs[k] = cur_Obj - sep_Obj + temp_Objs[k]
                                             
                        
                temp_max = np.argmax(temp_Objs)
                cur_Sol[cur_Ind] = temp_max
                cur_Obj = temp_Objs[temp_max]

        if itr == 0 or cur_Obj > best_Obj:
            best_Sol = np.array(cur_Sol)
            best_Obj = cur_Obj
            
    return best_Sol, best_Obj


if __name__ == "__main__":

    sim = -np.ones((9,9))
    sim[0:3,0:3] = +1
    sim[3:6,3:6] = +1
    sim[6:9,6:9] = +1
    
    lables = np.array([0,0,0,1,1,1,2,2,2])

    sol1 = max_correlation(sim, 3, 5)


def compute_consensus_map(cluster_ensemble, k, iter):
    N = len(cluster_ensemble[0])
    corr_graph = np.zeros((N, N))
    for encoder in cluster_ensemble:
        ss = encoder.argmax(axis=1)
        for i in range(0, N):
            for j in range(0, i):
                if ss[i] == ss[j]:
                    corr_graph[i, j] = corr_graph[i, j] + 1
                    corr_graph[j, i] = corr_graph[j, i] + 1
                else:
                    corr_graph[i, j] = corr_graph[i, j] - 1
                    corr_graph[j, i] = corr_graph[j, i] - 1

    consensus, loss = max_correlation(corr_graph, k, iter)

    return consensus