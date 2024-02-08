import copy
import numpy as np
import unittest
import random
import sys
sys.path.insert(0, '../')
from graph_transformer import *
random.seed(82729)
np.random.seed(38739)


# Gaussian elimination on W-matrix
def gauss_elim(m):
    nr, nc = m.shape
    lead = 0
    for r in range(nr):
        if nc <= lead:
            return m
        i = r
        while m[i, lead] == 0:
            i = i + 1
            if nr == i:
                i = r
                lead = lead + 1
                if nc == lead:
                    return m
        if i != r:  # swap rows
            h = copy.copy(m[i, :])
            m[i, :] = m[r, :]
            m[r, :] = h
        for i in range(r + 1, nr):
            if m[i, lead] != 0:
                m[i, :] = np.logical_xor(m[i, :], m[r, :])
        lead = lead + 1
    return m


# bring W-matrix in canonical form after Gaussian elimination has been applied
def get_canonical(m):
    nr, nc = m.shape
    first_one = np.zeros(nc, dtype=int)  # store where the row has its first one
    for r in range(nr):
        c = 0
        while c < nc:
            if m[r, c]:
                first_one[c] = r
                break
            c = c + 1
    for r in range(nr - 1):
        for c in range(r + 1, nc):
            if m[r, c] and first_one[c]:
                m[r, :] = np.logical_xor(m[r, :], m[first_one[c], :])
    return m


# change W-matrix with Q, H, R gate
def update_w_matrix(w_matrix, gate, n, num_nodes):
    if gate == 'Q':
        w_matrix[:, n] = np.logical_xor(w_matrix[:, n], w_matrix[:, n + num_nodes])
    elif gate == 'H':
        h = copy.copy(w_matrix[:, n])
        w_matrix[:, n] = w_matrix[:, n + num_nodes]
        w_matrix[:, n + num_nodes] = h
    elif gate == 'R':
        w_matrix[:, n + num_nodes] = np.logical_xor(w_matrix[:, n], w_matrix[:, n + num_nodes])


# get W-matrix representing the stabilizers
def get_w_matrix(g):
    w_matrix = np.concatenate((np.identity(g.number_of_nodes(), dtype=bool), nx.to_numpy_array(g, dtype=bool)), axis=1)
    l_nodes = list(g.nodes)  # same node labels can be missing due to fusion/measurement
    for n in range(g.number_of_nodes()):
        node = l_nodes[n]
        if len(g.nodes[node]['LC']) >= 1:
            update_w_matrix(w_matrix, g.nodes[node]['LC'][-1], n, g.number_of_nodes())
        if len(g.nodes[node]['LC']) == 2:
            update_w_matrix(w_matrix, g.nodes[node]['LC'][0], n, g.number_of_nodes())
    return w_matrix


# returns True if same edges (networkx puts the node added first to the beginning, so (0,1) or (1,0) is both possible)
def identical_edges(set1, set2):
    for e in set1.difference(set2):
        if not (e[1], e[0]) in set2:
            return False
    for e in set2.difference(set1):
        if not (e[1], e[0]) in set1:
            return False
    return True


# check local equivalence (can fail to recognize equivalence due to limited depth (depth=#qubits to explore full orbit))
def check_equivalence(g1, g2, depth=5):
    for n1 in g1.nodes:
        g1_cp = copy.deepcopy(g1)
        local_compl(g1_cp, n1)
        same = identical_edges(set(g1_cp.edges), set(g2.edges))
        if same:
            return True
        elif depth > 0:
            same = check_equivalence(g1_cp, g2, depth=depth-1)
        if same:
            return True
    return False


class TestFusionRules(unittest.TestCase):
    def single_qbt_test(self):
        # test Y-measurement
        for _ in range(20):
            gr = nx.gnp_random_graph(7, 0.5)
            nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
            gr2 = copy.deepcopy(gr)
            n1 = list(gr.nodes)[0]
            measure_y(gr, n1)
            measure_y_alternative(gr2, n1)
            self.assertTrue(identical_edges(set(gr.edges), set(gr2.edges)))  # the two implementations are equivalent
            self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

        # test X-measurement
        for _ in range(20):
            gr = nx.gnp_random_graph(7, 0.5)
            nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
            gr2 = copy.deepcopy(gr)
            n1 = list(gr.nodes)[0]
            measure_x(gr, n1)
            measure_x_alternative(gr2, n1)
            self.assertTrue(identical_edges(set(gr.edges), set(gr2.edges)))  # the two implementations are equivalent
            self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

        # Different special neighbor gives same stabilizer state (locally equivalent, H on different qubit)
        for _ in range(20):
            gr = nx.gnp_random_graph(7, 0.5)
            nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
            gr2 = copy.deepcopy(gr)
            n1 = list(gr.nodes)[0]
            measure_x(gr, n1, rnd=True)  # pick special neighbor randomly
            measure_x_alternative(gr2, n1, rnd=True)  # pick special neighbor randomly
            self.assertTrue(check_equivalence(gr, gr2))  # locally equivalent but not identical when special neighbor differs
            self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

        # test many single-qubit measurements
        for _ in range(20):
            gr = nx.gnp_random_graph(7, 0.5)
            nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
            gr2 = copy.deepcopy(gr)
            n1 = list(gr.nodes)[0]
            n2 = list(gr.nodes)[1]
            n3 = list(gr.nodes)[2]
            measure_single(gr, n1, 'X', method=0)
            measure_single(gr, n2, 'Y', method=0)
            measure_single(gr, n3, 'Z', method=0)
            measure_single(gr2, n1, 'X', method=1)
            measure_single(gr2, n2, 'Y', method=1)
            measure_single(gr2, n3, 'Z', method=1)
            self.assertTrue(identical_edges(set(gr.edges), set(gr2.edges)))  # the two implementations are equivalent
            self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

    def xzzx_test(self):
        qbt1 = 14
        qbt2 = 15
        sp1 = 13
        sp2 = 16
        gr = get_bcc_2d_lattice(5, 6)
        gr.remove_edge(qbt1, qbt2)
        measure_x(gr, qbt1, sp=sp1)
        measure_x(gr, qbt2, sp=sp2)
        gr2 = get_bcc_2d_lattice(5, 6)
        transform_xzzx(gr2, qbt1, qbt2, a_sp=sp1, b_sp=sp2)
        self.assertTrue(identical_edges(set(gr.edges), set(gr2.edges)))
        self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

        qbt1 = 2
        qbt2 = 7
        sp1 = 1
        sp2 = 6
        gr = get_double_chain(5)
        measure_x(gr, qbt1, sp=sp1)
        measure_x(gr, qbt2, sp=sp2)
        gr2 = get_double_chain(5)
        gr2.add_edge(qbt1, qbt2)
        transform_xzzx(gr2, qbt1, qbt2, a_sp=sp1, b_sp=sp2)
        self.assertTrue(identical_edges(set(gr.edges), set(gr2.edges)))

        gr = get_double_chain(5)
        transform_xzzx(gr, 2, 7)
        self.assertTrue(identical_edges(set(gr.edges), set([(0, 1), (5, 6), (3, 4), (8, 9), (1, 6), (1, 8), (3, 6), (3, 8)])))

        # test xzzx by comparison to circuit interpretation
        l_size = [6, 7, 8, 9, 10]
        for size in l_size:
            for _ in range(100):
                gr = nx.gnp_random_graph(size, 0.5)
                nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
                gr2 = copy.deepcopy(gr)
                n1 = list(gr.nodes)[0]
                n2 = list(gr.nodes)[1]
                transform_xzzx(gr, n1, n2)
                if (n1, n2) in gr2.edges:
                    gr2.remove_edge(n1, n2)
                else:
                    gr2.add_edge(n1, n2)
                measure_single(gr2, n1, 'X')
                measure_single(gr2, n2, 'X')
                self.assertTrue(check_equivalence(gr, gr2))  # a_* can be differently selected (edges can differ)
                self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

    def xyyx_test(self):
        gr = get_double_chain(5)
        transform_xyyx(gr, 2, 7)
        self.assertTrue(identical_edges(set(gr.edges), set([(0, 1), (5, 6), (3, 4), (8, 9), (1, 6), (1, 8), (3, 6), (3, 8), (1, 3), (6, 8)])))

        # test xyyx by comparison to circuit interpretation
        l_size = [6, 7, 8, 9, 10]
        for size in l_size:
            for _ in range(100):
                gr = nx.gnp_random_graph(size, 0.5)
                nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
                n1 = list(gr.nodes)[0]
                n2 = list(gr.nodes)[1]
                if (n1, n2) in gr.edges:
                    continue
                else:
                    gr2 = copy.deepcopy(gr)
                    for nb in set([n for n in gr2.neighbors(n1)]).symmetric_difference(set([n for n in gr2.neighbors(n2)])):
                        update_lc(gr2, nb, 'R')
                    local_compl(gr2, n1)
                    local_compl(gr2, n2)
                    gr2.add_edge(n1, n2)
                    measure_single(gr2, n1, 'X')
                    measure_single(gr2, n2, 'X')
                transform_xyyx(gr, n1, n2)
                self.assertTrue(check_equivalence(gr, gr2))
                self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

    def xyyz_test(self):
        gr = get_double_chain(5)
        transform_xyyz(gr, 2, 7)
        self.assertTrue(identical_edges(set(gr.edges), set([(0, 1), (5, 6), (3, 4), (8, 9), (1, 6), (1, 8), (3, 6), (3, 8), (6, 8)])))

    def xxzz_test(self):
        gr = get_double_chain(5)
        transform_xxzz(gr, 2, 7, sp=6)
        self.assertTrue(identical_edges(set(gr.edges), set([(0, 1), (3, 4), (8, 9), (1, 6), (3, 6), (6, 8), (1, 5), (3, 5), (5, 8)])))

    def yzzy_test(self):
        gr = get_double_chain(5)
        transform_yzzy(gr, 2, 7, sp=6)
        self.assertTrue(identical_edges(set(gr.edges), set([(0, 1), (3, 4), (8, 9), (1, 6), (3, 6), (6, 8), (1, 5), (3, 5), (5, 8), (1, 3)])))

        # test yzzy by comparison to circuit interpretation
        l_size = [6, 7, 8, 9, 10]
        for size in l_size:
            for _ in range(100):
                gr = nx.gnp_random_graph(size, 0.5)
                nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
                gr2 = copy.deepcopy(gr)
                n1 = list(gr.nodes)[0]
                n2 = list(gr.nodes)[1]
                transform_yzzy(gr, n1, n2)
                if (n1, n2) in gr2.edges:
                    gr2.remove_edge(n1, n2)
                else:
                    gr2.add_edge(n1, n2)
                measure_single(gr2, n1, 'Y')
                measure_single(gr2, n2, 'Y')  # second measurement can be changed by byproduct from first
                self.assertTrue(check_equivalence(gr, gr2))  # edges can differ as 2nd measurement can change to X
                self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())

    def deterministic_test(self):
        gr = nx.Graph()
        gr.add_nodes_from([0, 1, 2, 3])
        gr.add_edge(0, 1)
        gr.add_edge(1, 2)
        gr.add_edge(1, 3)
        nx.set_node_attributes(gr, '', 'LC')
        gr2 = copy.deepcopy(gr)
        gr3 = copy.deepcopy(gr)
        self.assertTrue(measure_double_parity(gr, 0, 1, 'XZZX'))
        self.assertTrue(measure_double_parity(gr2, 0, 1, 'YYZX'))
        self.assertFalse(measure_double_parity(gr3, 0, 1, 'XXZZ'))

        # shared neighborhood (unconnected fusion qubits)
        gr = nx.Graph()
        gr.add_nodes_from([0, 1, 2, 3])
        gr.add_edge(0, 2)
        gr.add_edge(0, 3)
        gr.add_edge(1, 2)
        gr.add_edge(1, 3)
        nx.set_node_attributes(gr, '', 'LC')
        gr2 = copy.deepcopy(gr)
        self.assertTrue(measure_double_parity(gr, 0, 1, 'XXZZ'))
        self.assertFalse(measure_double_parity(gr2, 0, 1, 'ZZYX'))

        # shared neighborhood (connected fusion qubits)
        gr = nx.Graph()
        gr.add_nodes_from([0, 1, 2, 3])
        gr.add_edge(0, 1)
        gr.add_edge(0, 2)
        gr.add_edge(0, 3)
        gr.add_edge(1, 2)
        gr.add_edge(1, 3)
        nx.set_node_attributes(gr, '', 'LC')
        gr2 = copy.deepcopy(gr)
        self.assertTrue(measure_double_parity(gr, 0, 1, 'YYXZ'))
        self.assertFalse(measure_double_parity(gr2, 0, 1, 'XXYZ'))

    # consistency check by different order of the fusion
    def consistency_test(self):
        l_size = [6, 7, 8, 9, 10]
        l_type = ['YZZY', 'XXZZ', 'XZZX', 'XYYX', 'XYYZ', 'YXZY']
        for fusion_type_1 in l_type:
            for fusion_type_2 in l_type:
                for size in l_size:
                    for _ in range(10):
                        gr = nx.gnp_random_graph(size, 0.5)
                        nx.set_node_attributes(gr, '', 'LC')  # store local Clifford (LC) gates as node attribute
                        gr2 = copy.deepcopy(gr)
                        l_nodes = list(gr.nodes)
                        measure_double_parity(gr, l_nodes[0], l_nodes[1], fusion_type_1)
                        measure_double_parity(gr, l_nodes[2], l_nodes[3], fusion_type_2)
                        measure_double_parity(gr2, l_nodes[2], l_nodes[3], fusion_type_2)
                        measure_double_parity(gr2, l_nodes[0], l_nodes[1], fusion_type_1)
                        self.assertTrue(check_equivalence(gr, gr2, depth=6))
                        self.assertTrue((get_canonical(gauss_elim(get_w_matrix(gr))) == get_canonical(gauss_elim(get_w_matrix(gr2)))).all())


if __name__ == '__main__':
    t = TestFusionRules()
    t.single_qbt_test()
    t.xzzx_test()
    t.xyyx_test()
    t.xyyz_test()
    t.xxzz_test()
    t.yzzy_test()
    t.consistency_test()
    t.deterministic_test()
