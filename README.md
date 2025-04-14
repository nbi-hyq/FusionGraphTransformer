# FusionGraphTransformer

This is a ```Python```-implementation for simulating the effect of linear-optics fusions (Bell state measurements) on graph states. Upon success, the considered type-II fusions represent a simultaneous measurement of two weight-two operators from the Pauli group. A detailed explanation for what this source code can be used is given in https://arxiv.org/abs/2405.02414.

see also:
- more information on graph states: https://arxiv.org/abs/quant-ph/0602096
- more information on linear-optics fusions: https://doi.org/10.25560/43936

## features
- simulate the effect of fusions on graph states (up to stabilizer signs).
- keep track of local Clifford byproduct operators (which allows to represent stabilizer states)
- simulate the effect of single-qubit measurements on graph states.
- visualize the graph states as a ```networkx``` graph
- automated tests and self-consistency checks

## source code
All the graph transformations are implemented in ```graph_transformer.py```. The graph (state) transformations upon a successful fusion are implemented in the functions ```transform_xzzx, transform_xxzz, transform_yzzy, transform_xyyx, transform_xyyz``` corresponding to the five fusion types described in https://arxiv.org/abs/2405.02414. Since fusions are measurements, their order is irrelevant. In ```tests/unit_tests.py```, we use this fact for several self-consistency tests. Modify a single line in the functions that implement the graph transformation and the test should likely give an error.

