######################################################
#####Functions
#Counting algorithm
Counting_Burnside <- function(n, k, B_list, N_list){
  S_list = NULL
  res = 1/k
  for(i in 1:(n-1)){
    B = B_list[i]
    N = N_list[i]
    x = rep(1, i+1)
    S = rep(0, B+N)
    for(l in 1:(B+N)){
      x = Burnside(i+1, k, x)
      cnt = length(which(x==x[i+1]))
      S[l] = (i+1)/(k*cnt)
    }
    res = res*mean(S[(B+1):(B+N)])
    S_list = append(S_list, list(S))
  }
  return(list(estimate = 1/res, stats = S_list))
}

#One iteration of the Burnside process
# x-starting state in [k]^n
# return y-new state in [k]^n
Burnside <- function(n, k, x){
  y = rep(0, n)
  for(l in 1:k){
    S = which(x==l)
    cycles = find_cycles(sample(length(S)))
    if(length(cycles)>0){
      for(i in 1:length(cycles)){
        coin = rmultinom(1, 1, rep(1/k,k))
        y[S[cycles[[i]]]] = which(coin == 1)
      }
    }
  }
  return(y)
}

#Decompose a permutation into cycles
find_cycles = function(perm){
  if(perm[1]==0){
    return(NULL)
  }
  n = length(perm)
  cycles = NULL
  count = 0
  visited = rep(0, n)
  while(count < n){
    head = which(visited == 0)[1]
    cycle = head
    count = count + 1
    visited[head] = 1
    current = perm[head]
    visited[current] = 1
    while(current != head){
      cycle = c(cycle, current)
      current = perm[current]
      count = count + 1
      visited[current] = 1
    }
    cycles = append(cycles, list(cycle))
  }
  return(cycles)
}

######################################################
#####Experiments
library(ggplot2)

Repeated_experiments <- function(n, k, B_list, N_list, n.replication){
  estimate = rep(0, n.replication)
  stats = NULL
  for(repli in 1:n.replication){
    print(repli)
    result = Counting_Burnside(n, k, B_list, N_list)
    estimate[repli] = result$estimate
    S = result$stats
    stats = append(stats, list(S))
  }
  return(list(estimate = estimate, stats = stats))
}

set.seed(0)
n = 20
estimate_list = rep(0, n)
true.val_list = rep(0, n)
B_list = rep(20, n-1)
N_list = rep(10000, n-1)
for(k in 2:n){
  print(k)
  result = Repeated_experiments(n, k, B_list, N_list, 1)
  estimate_list[k] = result$estimate
  true.val_list[k] = choose(n+k-1, k-1)
}
df = data.frame(k = 2:n, true = log(true.val_list[2:n]), estimate = log(estimate_list[2:n]))
ggplot(df, aes(x = k)) +
  geom_line(aes(y = estimate), color = "blue", linewidth = 0.5) +
  geom_point(aes(y = true, color = "True values", shape = "True values"), size = 2) +
  geom_point(aes(y = estimate, color = "Estimated values", shape = "Estimated values"), size = 3) +
  scale_color_manual(values = c("True values" = "red", "Estimated values" = "blue")) +
  scale_shape_manual(values = c("True values" = 8, "Estimated values" = 21)) + 
  labs(x = "k", y = "log(number of orbits)", color = NULL, shape = NULL, title = NULL) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 15),  
    axis.title.y = element_text(size = 15), 
    axis.text.x = element_text(size = 15),   
    axis.text.y = element_text(size = 15),  
    legend.text = element_text(size = 15)
  )

set.seed(0)
n = 20
B_list = rep(20, n-1)
N_list = rep(10000, n-1)
result_hist = Repeated_experiments(n, 10, B_list, N_list, 100)
estimate = log(result_hist$estimate)
k = 10
true = log(choose(n+k-1, k-1))
ggplot(data.frame(estimate), aes(x = estimate)) +
  geom_histogram(bins = 6, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = true, color = "True value"), linewidth = 0.8) +
  scale_color_manual(values = c("True value" = "red")) + 
  labs(title = NULL, x = "log(number of orbits)", y = "Frequency", color = NULL, linetype = NULL) +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 15),  
    axis.title.y = element_text(size = 15), 
    axis.text.x = element_text(size = 15),   
    axis.text.y = element_text(size = 15),   
    legend.text = element_text(size = 15)
  )
