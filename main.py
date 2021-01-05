import torch 
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")  
    print("Running on the GPU")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")
for i in range(10000):
	x=torch.randn((200,20))
	y=x**2+torch.randn((200,20))
	x=x.to(DEVICE)
	y=y.to(DEVICE)
	diff=torch.norm(y-x,dim=1)
	print(torch.mean(diff))


