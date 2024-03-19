# Plant Disease Model
A framework for running plant disease management optimisation experiments. 

## Installation
Plant disease model is based on the control_infra package. This is automatically installed by the supplied setup script. The source code and documentation are located here:  
https://github.com/racheltrimble/control_ablations_oc_paper

The optimal control code is based on casadi and requires the installation of the ma97 HSL linear solver. This can be installed by running the following the instructions here:  
https://github.com/casadi/casadi/wiki/Obtaining-HSL

The plant_disease_model/local_control_ablations_config.py file must them be updated to point to the installed library.  
    config.hsl_path = 'absolute_path_to_hsl_lib'

Then run setup.sh to create a local venv, install the required packages, create appropriate logging directories and install the code (as editable).

The data and charts for the papers that reference this repo can be replicated using the code in the notebooks directory (oc_vs_priority_based.py).

## Main Components
Plant disease model runs on the control_infra framework - a structure for running ablations and sweeps in computational testing of different control heuristics and algorithms. The functionality of plant disease model is split into four main components:
- simulator: A landscape level plant disease model including aerial dispersion and trade network spread.
- control: A series of targets built on the control_infra base classes to apply different control algorithms to the model.
- analysis: Code for analysing the data generated by the control targets built as a collection of control_infra plot_blocks.
- experiments: A series of studies for running sweeps and ablations on the model.

## Simulator interface
The epidemic model was wrapped to present a standard environment interface according to the Open AI gym framework. This library was designed for use with reinforcement learning algorithms but is convenient for evaluations of other mechanisms of control for stochastic systems. The main "step" function implemented by the environment takes the control signals or "action" as an argument and returns an observation describing the state of the system, a reward signal (equivalent to negative cost) and done. 
