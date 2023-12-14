import torch

edge_index = torch.LongTensor([[1,2,3,4],[4,3,2,1]])
print(edge_index)
node_feat = torch.randn((2,2,5,2))
print(node_feat)
edge_feat = torch.concat([node_feat[:,:,edge_index[0]],node_feat[:,:,edge_index[1]]],dim=-1)
print(edge_feat)
