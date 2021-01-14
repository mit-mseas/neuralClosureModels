# Neural Closure Models for Dynamical Systems

**Source code**: This directory contains the source code for Neural Closure Models for Dynamical Systems, available on arXiv: http://arxiv.org/abs/2012.13869

**Data sets**: Data is re-generated everytime before training for each of the experiments. However, all the runs used in the results and discussion section could be found here: http://mseas.mit.edu/download/guptaa/Neural_closure_models_paper_final_runs/

**Directory structure and scripts**

- `src/utilities/DDE_Solver.py`: A delay differential equation solver compatible with TensorFlow.
- `src/solvers/`: Contains all the functions/classes needed for adjoint equation and the main training loop for both discrete-DDE and distributed-DDE implementations
- `examples/`: Contains some examples demonstrating how to use the DDE solver and the neural-DDE implementations
- Experiments - 1 
	- `testcases/AD_Eqn_ROM/`: Contains the scripts corresponding to the different neural closure models
	- `src/advec_diff_case/rom_advec_diff.py`: Script containing the RHS of the POD-GP ROM for the Burger's equations
- Experiments - 2
	- `testcases/AD_Eqn_Res/`: Contains the scripts corresponding to the different neural closure models 
	- `src/advec_diff_case/advec_advec_diff.py`: Script containing the RHS of the FOM for the Burger's equations
- Experiments - 3 
	- `testcases/Bio_Eqn/`: Contains the scripts corresponding to the different neural closure models
	- `src/bio_eqn_case/bio_eqn.py`: Script containing the RHS of the NPZ and NNPZD models

### Abstract
Complex dynamical systems are used for predictions in many applications. Because of computational costs, models are however often truncated, coarsened, or aggregated. As the neglected and unresolved terms along with their interactions with the resolved ones become important, the usefulness of model predictions diminishes. We develop a novel, versatile, and rigorous methodology to learn non-Markovian closure parameterizations for low-fidelity models using data from high-fidelity simulations. The new *neural closure models* augment low-fidelity models with neural delay differential equations (nDDEs), motivated by the Mori-Zwanzig formulation and the inherent delays in natural dynamical systems. We demonstrate that neural closures efficiently account for truncated modes in reduced-order-models, capture the effects of subgrid-scale processes in coarse models, and augment the simplification of complex biochemical models. We show that using non-Markovian over Markovian closures improves long-term accuracy and requires smaller networks. We provide adjoint equation derivations and network architectures needed to efficiently implement the new discrete and distributed nDDEs. The performance of discrete over distributed delays in closure models is explained using information theory, and we observe an optimal amount of past information for a specified architecture. Finally, we analyze computational complexity and explain the limited additional cost due to neural closure models.


