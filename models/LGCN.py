import torch
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, scatter_


class LGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_nn, L=4, make_bidirectional=False, neighbor_nl=False, DVE=False):
        super().__init__(aggr='add')
        
        self.L = L
        self.DVE = DVE
        self.neighbor_nl = neighbor_nl
        self.edge_nn = edge_nn
        self.make_bidirectional = make_bidirectional
        
        if self.DVE:
            in_channels += 2*L
        self.in_channels = in_channels
        
        self.self_loop_weight = torch.nn.Parameter(torch.ones(L))
        self.lin = torch.nn.Linear(L*in_channels, out_channels)
        
        if self.neighbor_nl:
            self.lin = torch.nn.Linear(L*in_channels, 2*out_channels)
            self.lin2 = torch.nn.Linear(2*out_channels, out_channels)
        
        if self.make_bidirectional:
            self.lin = torch.nn.Linear(2*L*in_channels, out_channels)
            if self.neighbor_nl:
                self.lin = torch.nn.Linear(2*L*in_channels, 2*out_channels)

            self.padding_func1 = torch.nn.ConstantPad2d((0,L,0,0),0)
            self.padding_func2 = torch.nn.ConstantPad2d((L,0,0,0),0)
            
            
    def get_embeddings(self, edge_attr, edge_attr_cutoffs):
        # batches for increased performance and less GPU allocation
        batches = []
        for lims in edge_attr_cutoffs:
            if lims[1] == -1:
                batches.append(self.edge_nn(edge_attr[lims[0]:]))
            else:
                batches.append(self.edge_nn(edge_attr[lims[0]:lims[1],:,:lims[2]]))
        return torch.cat(batches)
    

    def forward(self, x, edge_index, edge_attr, edge_attr_cutoffs):

        # add on-vertex embeddings if required
        if self.DVE:
            row, col = edge_index
            edge_embeddings = self.get_embeddings(edge_attr,edge_attr_cutoffs)
            edge_embedding_collected1 = scatter_("mean",edge_embeddings,row)
            edge_embedding_collected2 = scatter_("mean",edge_embeddings,col)
                        
            # protection against last node in the list having only one associated edge direction
            shape_difference1 = x.shape[0]-edge_embedding_collected1.shape[0]
            if shape_difference1 > 0:
                padding_func = torch.nn.ConstantPad2d((0,0,0,shape_difference1),0)
                edge_embedding_collected1 = padding_func(edge_embedding_collected1)
                
            shape_difference2 = x.shape[0]-edge_embedding_collected2.shape[0]
            if shape_difference2 > 0:
                padding_func = torch.nn.ConstantPad2d((0,0,0,shape_difference2),0)
                edge_embedding_collected2 = padding_func(edge_embedding_collected2)
            
            x = torch.cat((x, edge_embedding_collected1, edge_embedding_collected2), 1)
        
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        
        if self.make_bidirectional:
            edge_index = torch.cat((edge_index,edge_index.flip(0)),dim=1)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr, edge_attr_cutoffs=edge_attr_cutoffs)
    

    def message(self, x_j, edge_index, size, edge_attr, edge_attr_cutoffs):

        if self.make_bidirectional:
            # back to unidirectionality for reduced learning time
            row, col = edge_index[:,:int(edge_index.shape[1]/2)]
        else:
            row, col = edge_index

        first_weights = self.get_embeddings(edge_attr,edge_attr_cutoffs)

        first_weights = first_weights.view(-1,self.L) #fix for L=1
        weights = self.self_loop_weight.repeat(row.size(0),1)
        weights[:first_weights.size(0)] = first_weights

        if self.make_bidirectional:
            # convert to bidirectional again
            weights = torch.cat((self.padding_func1(weights),self.padding_func2(weights)))
            row, col = edge_index
        
        deg_weighted = scatter_("add",weights,col)+1e-5
        deg_weighted_inv = 1/deg_weighted
        norm_weighted = deg_weighted_inv[col]
        
        embedding_matrix = torch.matmul(norm_weighted.view(row.size(0),-1,1),x_j.view(row.size(0),1,-1))
        embedding_flattened = embedding_matrix.view(row.size(0),-1)
        
        messages = self.lin(embedding_flattened)
        
        if self.neighbor_nl:
            messages = F.dropout(messages, p=0.2, training=self.training)
            messages = torch.tanh(messages)
            messages = self.lin2(messages)

        return messages
    
    
    def update(self, aggr_out):
        return aggr_out

