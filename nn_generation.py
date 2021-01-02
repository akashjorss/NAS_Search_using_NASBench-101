from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import sys, os
import numpy as np
import copy
import hashlib
import itertools
import math


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.conv_bn_relu(x)


class Conv3x3BnRelu(nn.Module):
    """3x3 convolution with batch norm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super(Conv3x3BnRelu, self).__init__()

        self.conv3x3 = ConvBnRelu(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv3x3(x)
        return x


class Conv1x1BnRelu(nn.Module):
    """1x1 convolution with batch norm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super(Conv1x1BnRelu, self).__init__()

        self.conv1x1 = ConvBnRelu(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class MaxPool3x3(nn.Module):
    """3x3 max pool with no subsampling."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MaxPool3x3, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        return x


# Commas should not be used in op names
OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3
}


def gen_is_edge_fn(bits):
    """Generate a boolean function for the edge connectivity.

  Given a bitstring FEDCBA and a 4x4 matrix, the generated matrix is
    [[0, A, B, D],
     [0, 0, C, E],
     [0, 0, 0, F],
     [0, 0, 0, 0]]

  Note that this function is agnostic to the actual matrix dimension due to
  order in which elements are filled out (column-major, starting from least
  significant bit). For example, the same FEDCBA bitstring (0-padded) on a 5x5
  matrix is
    [[0, A, B, D, 0],
     [0, 0, C, E, 0],
     [0, 0, 0, F, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]

  Args:
    bits: integer which will be interpreted as a bit mask.

  Returns:
    vectorized function that returns True when an edge is present.
  """

    def is_edge(x, y):
        """Is there an edge from x to y (0-indexed)?"""
        if x >= y:
            return 0
        # Map x, y to index into bit string
        index = x + (y * (y - 1) // 2)
        return (bits >> index) % 2 == 1

    return np.vectorize(is_edge)


def is_full_dag(matrix):
    """Full DAG == all vertices on a path from vert 0 to (V-1).

  i.e. no disconnected or "hanging" vertices.

  It is sufficient to check for:
    1) no rows of 0 except for row V-1 (only output vertex has no out-edges)
    2) no cols of 0 except for col 0 (only input vertex has no in-edges)

  Args:
    matrix: V x V upper-triangular adjacency matrix

  Returns:
    True if the there are no dangling vertices.
  """
    shape = np.shape(matrix)

    rows = matrix[:shape[0] - 1, :] == 0
    rows = np.all(rows, axis=1)  # Any row with all 0 will be True
    rows_bad = np.any(rows)

    cols = matrix[:, 1:] == 0
    cols = np.all(cols, axis=0)  # Any col with all 0 will be True
    cols_bad = np.any(cols)

    return (not rows_bad) and (not cols_bad)


def num_edges(matrix):
    """Computes number of edges in adjacency matrix."""
    return np.sum(matrix)


def hash_module(matrix, labeling):
    """Computes a graph-invariance MD5 hash of the matrix and label pair.

  Args:
    matrix: np.ndarray square upper-triangular adjacency matrix.
    labeling: list of int labels of length equal to both dimensions of
      matrix.

  Returns:
    MD5 hash of the matrix and labeling.
  """
    vertices = np.shape(matrix)[0]
    in_edges = np.sum(matrix, axis=0).tolist()
    out_edges = np.sum(matrix, axis=1).tolist()

    assert len(in_edges) == len(out_edges) == len(labeling)
    hashes = list(zip(out_edges, in_edges, labeling))
    hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
    # Computing this up to the diameter is probably sufficient but since the
    # operation is fast, it is okay to repeat more times.
    for _ in range(vertices):
        new_hashes = []
        for v in range(vertices):
            in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
            out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
            new_hashes.append(hashlib.md5(
                (''.join(sorted(in_neighbors)) + '|' +
                 ''.join(sorted(out_neighbors)) + '|' +
                 hashes[v]).encode('utf-8')).hexdigest())
        hashes = new_hashes
    fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

    return fingerprint


def permute_graph(graph, label, permutation):
    """Permutes the graph and labels based on permutation.

  Args:
    graph: np.ndarray adjacency matrix.
    label: list of labels of same length as graph dimensions.
    permutation: a permutation list of ints of same length as graph dimensions.

  Returns:
    np.ndarray where vertex permutation[v] is vertex v from the original graph
  """
    # vertex permutation[v] in new graph is vertex v in the old graph
    forward_perm = zip(permutation, list(range(len(permutation))))
    inverse_perm = [x[1] for x in sorted(forward_perm)]
    edge_fn = lambda x, y: graph[inverse_perm[x], inverse_perm[y]] == 1
    new_matrix = np.fromfunction(np.vectorize(edge_fn),
                                 (len(label), len(label)),
                                 dtype=np.int8)
    new_label = [label[inverse_perm[i]] for i in range(len(label))]
    return new_matrix, new_label


def is_isomorphic(graph1, graph2):
    """Exhaustively checks if 2 graphs are isomorphic."""
    matrix1, label1 = np.array(graph1[0]), graph1[1]
    matrix2, label2 = np.array(graph2[0]), graph2[1]
    assert np.shape(matrix1) == np.shape(matrix2)
    assert len(label1) == len(label2)

    vertices = np.shape(matrix1)[0]
    # Note: input and output in our constrained graphs always map to themselves
    # but this script does not enforce that.
    for perm in itertools.permutations(range(0, vertices)):
        pmatrix1, plabel1 = permute_graph(matrix1, label1, perm)
        if np.array_equal(pmatrix1, matrix2) and plabel1 == label2:
            return True

    return False


class ModelSpec(object):
    """Model specification given adjacency matrix and labeling."""

    def __init__(self, matrix, ops, data_format='channels_last'):
        """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        shape = np.shape(matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('matrix must be square')
        if shape[0] != len(ops):
            raise ValueError('length of ops must match matrix dimensions')
        if not is_upper_triangular(matrix):
            raise ValueError('matrix must be upper triangular')

        # Both the original and pruned matrices are deep copies of the matrix and
        # ops so any changes to those after initialization are not recognized by the
        # spec.
        self.original_matrix = copy.deepcopy(matrix)
        self.original_ops = copy.deepcopy(ops)

        self.matrix = copy.deepcopy(matrix)
        self.ops = copy.deepcopy(ops)
        self.valid_spec = True
        self._prune()

        self.data_format = data_format

    def _prune(self):
        """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
        num_vertices = np.shape(self.original_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if self.original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if self.original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            self.matrix = None
            self.ops = None
            self.valid_spec = False
            return

        self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
        self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del self.ops[index]

    def hash_spec(self, canonical_ops):
        """Computes the isomorphism-invariant graph hash of this spec.

    Args:
      canonical_ops: list of operations in the canonical ordering which they
        were assigned (i.e. the order provided in the config['available_ops']).

    Returns:
      MD5 hash of this spec which can be used to query the dataset.
    """
        # Invert the operations back to integer label indices used in graph gen.
        labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
        return hash_module(self.matrix, labeling)

    def visualize(self):
        """Creates a dot graph. Can be visualized in colab directly."""
        num_vertices = np.shape(self.matrix)[0]
        g = graphviz.Digraph()
        g.node(str(0), 'input')
        for v in range(1, num_vertices - 1):
            g.node(str(v), self.ops[v])
        g.node(str(num_vertices - 1), 'output')

        for src in range(num_vertices - 1):
            for dst in range(src + 1, num_vertices):
                if self.matrix[src, dst]:
                    g.edge(str(src), str(dst))

        return g


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


class Network(nn.Module):
    def __init__(self, spec, args):
        super(Network, self).__init__()

        self.layers = nn.ModuleList([])

        in_channels = args['in_channels']
        out_channels = args.stem_out_channels

        # initial stem convolution
        stem_conv = ConvBnRelu(in_channels, out_channels, 3, 1, 1)
        self.layers.append(stem_conv)

        in_channels = out_channels
        for stack_num in range(args.num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for module_num in range(args.num_modules_per_stack):
                cell = Cell(spec, in_channels, out_channels)
                self.layers.append(cell)
                in_channels = out_channels

        self.classifier = nn.Linear(out_channels, args.num_labels)

        self._initialize_weights()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        out = torch.mean(x, (2, 3))
        out = self.classifier(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Cell(nn.Module):
    """
    Builds the model using the adjacency matrix and op labels specified. Channels
    controls the module output channel count but the interior channels are
    determined via equally splitting the channel count whenever there is a
    concatenation of Tensors.
    """

    def __init__(self, spec, in_channels, out_channels):
        super(Cell, self).__init__()

        self.spec = spec
        self.num_vertices = np.shape(self.spec.matrix)[0]

        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = ComputeVertexChannels(in_channels, out_channels, self.spec.matrix)
        # self.vertex_channels = [in_channels] + [out_channels] * (self.num_vertices - 1)

        # operation for each node
        self.vertex_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices - 1):
            op = OP_MAP[spec.ops[t]](self.vertex_channels[t], self.vertex_channels[t])
            self.vertex_op.append(op)

        # operation for input on each vertex
        self.input_op = nn.ModuleList([None])
        for t in range(1, self.num_vertices):
            if self.spec.matrix[0, t]:
                self.input_op.append(Projection(in_channels, self.vertex_channels[t]))
            else:
                self.input_op.append(None)

    def forward(self, x):
        tensors = [x]

        out_concat = []
        for t in range(1, self.num_vertices - 1):
            fan_in = [Truncate(tensors[src], self.vertex_channels[t]) for src in range(1, t) if
                      self.spec.matrix[src, t]]

            if self.spec.matrix[0, t]:
                fan_in.append(self.input_op[t](x))

            # perform operation on node
            # vertex_input = torch.stack(fan_in, dim=0).sum(dim=0)
            vertex_input = sum(fan_in)
            # vertex_input = sum(fan_in) / len(fan_in)
            vertex_output = self.vertex_op[t](vertex_input)

            tensors.append(vertex_output)
            if self.spec.matrix[t, self.num_vertices - 1]:
                out_concat.append(tensors[t])

        if not out_concat:
            assert self.spec.matrix[0, self.num_vertices - 1]
            outputs = self.input_op[self.num_vertices - 1](tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.spec.matrix[0, self.num_vertices - 1]:
                outputs += self.input_op[self.num_vertices - 1](tensors[0])

            # if self.spec.matrix[0, self.num_vertices-1]:
            #    out_concat.append(self.input_op[self.num_vertices-1](tensors[0]))
            # outputs = sum(out_concat) / len(out_concat)

        return outputs


def Projection(in_channels, out_channels):
    """1x1 projection (as in ResNet) followed by batch normalization and ReLU."""
    return ConvBnRelu(in_channels, out_channels, 1)


def Truncate(inputs, channels):
    """Slice the inputs to channels if necessary."""
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs  # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]


def ComputeVertexChannels(in_channels, out_channels, matrix):
    """Computes the number of channels at every vertex.

    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.

    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


class CustomDict(dict):
    def __init__(self):
        super().__init__()

    def __getattr__(self, key):
        return self.get(key)


def generate_net(spec, args):
    return Network(spec, args)


if __name__ == '__main__':
    matrix = [[0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0]]
    operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3',
                  'output']  # How many channels in 1x1 filter?

    # parser = argparse.ArgumentParser(description='NASBench')
    # parser.add_argument('--module_vertices', default=7, type=int, help='#vertices in graph')
    # parser.add_argument('--max_edges', default=9, type=int, help='max edges in graph')
    # parser.add_argument('--available_ops', default=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
    #                     type=list, help='available operations performed on vertex')
    # parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    # parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    # parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
    # parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    # parser.add_argument('--epochs', default=100, type=int, help='#epochs of training')
    # parser.add_argument('--learning_rate', default=0.025, type=float, help='base learning rate')
    # parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
    # parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')
    # parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
    # parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
    # parser.add_argument('--num_labels', default=10, type=int, help='#classes')
    #
    # args = parser.parse_args()

    args = CustomDict()
    args['module_vertices'] = 7
    args['max_edges'] = 9
    args['available_ops'] = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
    args['stem_out_channels'] = 128
    args['num_stacks'] = 3
    args['num_modules_per_stack'] = 3
    args['batch_size'] = 128
    args['epochs'] = 100
    args['learning_rate'] = 0.025
    args['lr_decay_method'] = 'COSINE_BY_STEP'
    args['momentum'] = 0.9
    args['weight_decay'] = 1e-4
    args['grad_clip'] = 5
    args['num_labels'] = 10

    spec = ModelSpec(matrix, operations)
    net = generate_net(spec, args)
    print(net)
    print(args.num_stacks)

    # myDict = CustomDict()
    # myDict['akash'] = 26
    # myDict['snigdha'] = 31
    # print(myDict.get('akash'))
    # print(myDict.akash)
