# AdaptiveStressTesting

This is a python (3.5.2) implementation of the AdaptiveStressTesting of the Julia package [AdaptiveStressTesting.jl](https://github.com/sisl/AdaptiveStressTesting.jl).

# Usage

The usage of the package is similiar to its orginal Julia version. User needs to provide a ``sim`` object with three functions:

* ``initialize(sim)`` - Resets the simulator to the initial state
* ``update(sim)`` - Steps the simulator forward a single time step.  The tuple ``(prob, isevent, dist)`` is returned, where prob is the probability of taking that step, isevent indicates whether the failure event has occurred, and dist is an optional distance metric that hints to the optimizer how close the execution was to an event.
* ``isterminal(sim)`` - Returns true if the simulation has ended, false otherwise. 

These functions, along with configuration parameters, should be passed to create the adaptive stress test object 
```
ast = AdaptiveStressTesting.AdaptiveStressTest(ast_params, sim, MySimType.initialize, MySimType.update, MySimType.isterminal)
```

To draw Monte Carlo samples from the simulator:
```
ASTSim.sample(ast)
```

To run the stess test:
```
result = AST_MCTS.stress_test(ast, mcts_params)
```
where ``mcts_params`` is a ``DPWParams`` object containing the Monte Carlo tree search parameters.
The result object contains the total reward, action sequence, and q-values of the found execution path of the simulator.
```
result.reward
result.action_seq
result.q_values
```
