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
- Get Brax up and running (i.e. ingesting URDF, setting simulation state, setting URDF parameters programatically, stepping simulation, backpropagating state error to parameters)
- Be able to select which parameters are backpropagated to (if possible)
- Setup naive optimization method and interface to start testing with MuJoCo
    - Make the optimization method easily swappable (.json config system would likely be best)
- Get MuJoCo up and running (selectable policy) and sending data to identification module as well as simulating (togglable)
- Use identification module outputs to update MuJoCo URDF parameters on-the-fly (ideally, those changes should propagate to MPC as well)
- Set up automatic experiment running (experiment configs that include what mujoco should simulate, what policy to use in mujoco, what optimization method to use in Brax, where to log results, how often to update URDF parameters for policy, how often to send data from MuJoCo to Brax, etc.)