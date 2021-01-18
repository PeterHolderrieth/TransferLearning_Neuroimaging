import torch 
import torch.nn as nn

gamma=0.3
model=nn.Sequential(nn.BatchNorm1d(1,momentum=gamma))

model.train()
n_steps=100
running_mean=0.

#The model initializes the scaling parameters by offset=0 and scaling=1
print(running_mean)

for i in range(n_steps):
    #The running mean is initi
    if torch.abs(model[0].running_mean-running_mean)>1e-6:
        print("Error.")
    x=torch.tensor([3,4,5,6],dtype=torch.float32).unsqueeze(1)
    #The momentum is the share of new parameters being added:
    running_mean=gamma*x.mean()+(1-gamma)*running_mean
    #Compute the output:
    out=model(x)
    #Compute using running averages:
    out_running=(x-model[0].running_mean)/(torch.sqrt(model[0].running_var+1e-5))
    #Compute using batch averages:
    out_batch=(x-4.5)/(torch.sqrt(x.var(unbiased=False)+1e-5))

    print("Running:", out_running)
    print("Batch: ", out_batch)
    print("Pytorch: ", out)

#- The batch normalization is done by old_average*(1-momentum)+batch_average*momentum
#--> default value 0.1 seems reasonable
#- The variation uses the biased estimator. Why??? Wouldn't it be better 
#to use the unbiased estimator, especially for small batch sizes, that 
#seems much more reasonable!
#Why is the running variance using the unbiased estimator?!
#My answer: the biased estimator has a lower MSE (lowest actually for scaling 
# factor 1/(n+1)). Therefore, it makes sense to get the estimator with the lowest MSE
#for small batch sizes.
#For the running average, we try to avoid any bias since the law of large numbers hold.
#Therefore, we use the unbiased estimator of the variance. This causes the results above to 
#be different.

print()
print()
print()
gamma=0.3
model.eval()
n_steps=10
running_mean=0.
#Compute the output:
out=model(x)
#Compute using running averages:
out_running=(x-model[0].running_mean)/(torch.sqrt(model[0].running_var+1e-5))
#Compute using batch averages:
out_batch=(x-4.5)/(torch.sqrt(x.var(unbiased=False)+1e-5))

print("Running:", out_running)
print("Batch: ", out_batch)
print("Pytorch: ", out)
#During evaluation, we see that they use running average.

