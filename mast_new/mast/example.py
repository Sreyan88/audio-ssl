import torch

centroids = torch.load('tensor_data.pt')
print(torch.argmin(centroids.norm(dim=-1)))
centroids = centroids/centroids.norm(dim=-1, keepdim=True)
memory_bank = [torch.rand(128, 512) for i in range(1024)]
print(memory_bank[0])
print(memory_bank[1])
memory_avg = [memory_bank[i].T.mean(dim=0) for i in range(len(memory_bank))]
memory_avg = torch.stack(memory_avg)
memory_avg = torch.rand(1024,128)
print('memory_avg', memory_avg.norm(dim=-1).shape)
memory_avg = memory_avg/memory_avg.norm(dim=-1,keepdim=True)
print('memory_avg', memory_avg.shape)
centroid_pairwise_dist = torch.topk(torch.torch.cdist(centroids, centroids, p=2),k=len(centroids),dim=1).indices
print('centroid_pairwise', centroid_pairwise_dist.shape)
memory_centroid_dist = torch.argmin(torch.cdist(memory_avg, centroids, p=2), dim=1)
print('memory_centroid_dist', memory_centroid_dist.shape)
x = torch.rand(128, 512)
print(x.T.mean(dim=0).shape)
point_cluster = torch.argmin(torch.cdist(x.T.mean(dim=0).unsqueeze(0), centroids, p=2), dim=1)

print(point_cluster)
print(memory_centroid_dist)
l = []
for i in range(centroid_pairwise_dist.shape[1]):
    flag = 0
    for j in range(memory_centroid_dist.shape[0]):
        if centroid_pairwise_dist[point_cluster.tolist()[0]][i] ==  memory_centroid_dist[j]:
            l.append(j)
            flag = 1
    if flag == 1:
        break       

print(l)