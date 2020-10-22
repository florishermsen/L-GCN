import torch
from torch.nn import Parameter, Linear, ConstantPad2d
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, scatter_


class LGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_network, *, L=4,
                 make_bidirectional=False, neighbor_nl=False):

        # add, because we do our own normalization
        super().__init__(aggr='add')
        
        self.L = L
        self.neighbor_nl = neighbor_nl
        self.edge_network = edge_network
        self.make_bidirectional = make_bidirectional
        
        self.in_channels = in_channels
        
        self.self_loop_weight = Parameter(torch.ones(L))
        self.lin = Linear(L*in_channels, out_channels)
        
        # two linear layers instead of one in case of extra per-message
        #   nonlinearity
        if self.neighbor_nl:
            self.lin = Linear(L * in_channels, 2 * out_channels)
            self.lin2 = Linear(2 * out_channels, out_channels)
        
        # expansion of in-channels in case of bidirectionality
        #   introduced inside the architecture itself
        if self.make_bidirectional:
            self.lin = Linear(2 * L * in_channels, out_channels)
            if self.neighbor_nl:
                # only need to redefine first layer,
                #   second remains the same
                self.lin = Linear(2 * L * in_channels, 2 * out_channels)

            # padding functions for the latent representations
            #   zeros appended to original, zeros prepended to
            #   carbon copy in other direction.
            self.padding_func1 = ConstantPad2d((0, L, 0, 0), 0)
            self.padding_func2 = ConstantPad2d((L, 0, 0, 0), 0)


    def get_embeddings(self, edge_attr, edge_attr_cutoffs):
        if edge_attr_cutoffs is not None:
            # batches for increased performance in case of irregular
            #   number of edges per node pair.
            #   sorted and grouped by original size of multi-edge
            #   population to reduce memory allocation and increase
            #   training speed.
            batches = []
            for lims in edge_attr_cutoffs:
                if lims[1] == -1:
                    batches.append(
                        self.edge_network(
                            edge_attr[lims[0]:]
                        )
                    )
                else:
                    batches.append(
                        self.edge_network(
                            edge_attr[lims[0]:lims[1], :, :lims[2]]
                        )
                    )
            return torch.cat(batches)
        else:
            return self.edge_network(edge_attr)
            

    def forward(self, x, edge_index, edge_attr, *,
                edge_attr_cutoffs=None):

        # add self-loops to allow retainment of properties
        edge_index, _ = add_self_loops(
            edge_index,
            num_nodes=x.size(0)
        )

        if self.make_bidirectional:
            # flip indices and append to introduce bidirectional
            #   propagation. Note that this does require different
            #   treatment, see padding functions and expansion of
            #   latent representations
            edge_index = torch.cat(
                (
                    edge_index,
                    edge_index.flip(0)
                ),
                dim=1
            )

        return self.propagate(
            edge_index,
            size=(x.size(0), x.size(0)),
            x=x,
            edge_attr=edge_attr,
            edge_attr_cutoffs=edge_attr_cutoffs
        )
    
    
    def message(self, x_j, edge_index_i, edge_index_j, edge_attr,
                edge_attr_cutoffs):
        
        row, col = edge_index_j, edge_index_i

        if self.make_bidirectional:
            cutoff = int(row.shape[0] / 2)
            row, col = row[:cutoff], col[:cutoff]

        first_weights = self.get_embeddings(
            edge_attr,
            edge_attr_cutoffs
        )

        # fix for L=1:
        first_weights = first_weights.view(-1, self.L)

        weights = self.self_loop_weight.repeat(row.size(0), 1)
        # because self-loops (without edge features) are appended
        weights[:first_weights.size(0)] = first_weights

        if self.make_bidirectional:
            # convert to bidirectional
            weights = torch.cat(
                (
                    self.padding_func1(weights),
                    self.padding_func2(weights)
                )
            )
            row, col = edge_index_j, edge_index_i
        
        # epsilon due to upcoming division
        deg_weighted = scatter_("add",weights,col) + 1e-5 
        deg_weighted_inv = 1 / deg_weighted
        norm_weighted = deg_weighted_inv[col]
        
        # make tensor product of (latent) edge features and node
        #   features
        embedding_matrix = torch.matmul(
            norm_weighted.view(row.size(0), -1, 1),
            x_j.view(row.size(0), 1, -1)
        )
        # vectorize:
        embedding_flattened = embedding_matrix.view(row.size(0), -1)
        
        messages = self.lin(embedding_flattened)
        
        # more layers in case of L-GCN+ version
        #   allows for additional nonlinarity on a per-message
        #   / per-neighbor basis
        if self.neighbor_nl:
            messages = F.dropout(
                messages,
                p=0.2,
                training=self.training
            )
            messages = torch.tanh(messages)
            messages = self.lin2(messages)

        return messages
    

    def update(self, aggr_out):
        # aggregate and return result
        return aggr_out
