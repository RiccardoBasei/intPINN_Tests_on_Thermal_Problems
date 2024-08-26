# intPINN_Tests_on_Thermal_Problems
Development of a PINN-based model of the thermal behaviour of two electromagnetic devices. These devices are represented as first-order dynamic systems after applying a FEM discretization (and a POD reduction). Considering the input variable $\mathbf{u}$ constant in the interval of interest, the PINN $f$ learns the average value of the derivative of the state vector $\mathbf{x}$ in that interval, useful to determine the state at the end of the interval $\mathbf{x}_{t+\Delta t}$ with the following equation:

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t+\Delta tf(\Delta t, \mathbf{u}_t, \mathbf{x}_t)$$

where $\Delta t$ is the length of the interval.

The devices under analysis are a Chip with heat sink (*ROM3*, *ROM10*, *ROM3_non_lin*) and a MOSFET (*ROM32_non_lin*).

## Folders content
* in `create_parameters` the main parameters are defined
* in `custom_resnet_parallel_training` the network is defined and the training loop is implemented
* in `dataset_generation`* the datasets cointing collocation points are built
* in `training` all the steps useful to train the PINN are executed sequentially
* in `test` the model is tested for an input variable time evolution
