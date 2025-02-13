# URDFit - Fitting URDF Parameters via Differentiable Simulation
## NOTE: URDFit is a WORK IN PROGRESS


### Design
- Interface:
    - Single class
    - path to a URDF is given as a source
    - other hyperparameters passed in init
    - data is given to the class in the form (urdf_parameters, start_state, action, end_state) (should accept batched input)
        - class then takes the state, sets sim to that state, sets URDF parameters to given parameters, simulates for one time step, backpropagates state error to parameters, returns optimal parameters
        - exactly how to compute optimal parameters remains TBD, but initially just get naive gradient descent done


### TODO:
- Setup naive optimization method and interface to start testing with MuJoCo
- Get MuJoCo up and running (selectable policy) and sending data to identification module as well as simulating (togglable)
- Use identification module outputs to update MuJoCo URDF parameters on-the-fly (ideally, those changes should propagate to MPC as well)
- Set up automatic experiment running (experiment configs that include what mujoco should simulate, what policy to use in mujoco, what optimization method to use in Brax, where to log results, how often to update URDF parameters for policy, how often to send data from MuJoCo to Brax, etc.)
- Set up a script which "scrambles" URDF parameters according to some type of randomness (random offset sampled from some distribution type, completely random values, swapping values around, whatever)

MuJoCo stuff is just for myself, not for URDFit. URDFit is only a class that has hyperparameters related to system identification.
The MuJoCo stuff does not need to be a whole fancy library, any extra code I write should just be for making experimentation faster and easier.

Experimentation workflow:
- set up config.json for the experiment (only contains URDFIT parameters)
- run mj_experiment.py -c <path_to_config_file> --xml <path_to_mujoco_mpc_xml> --urdf <path_to_urdf_with_inaccurate_parameters>
    - other possible flags --visualize, --eval <eval_dir>, --policy-type <type>, --trajectory-batch-size <how_many_steps_between_URDF_updates>, --policy-update-freq, etc.
- maybe instead of all these arguments, make it a large experiment config file?

Overall: urdfit will be self-contained python library

experiment script can be one or a few python scripts in experiment folder (no need to make a package)


Installation (Windows) (other platforms might require different installation steps):

`conda create -n urdfit-env python=3.11`

`conda activate urdfit-env`

`pip install -r requirements.txt`

`pip install -e .`

NOTE: Only accepts MJCF files, but you can use `urdf2mjcf path/to/your/robot.urdf` to convert a URDF to MJCF format. (already included in requirements.txt)