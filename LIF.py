"""
Leaky integrate and fire (LIF) neuron population with decaying-exponential post-synaptic current
"""


import pyNN.spiNNaker as sim
from pyNN.space import RandomStructure, Sphere
import numpy as np
import matplotlib.pyplot as plt
from spynnaker.pyNN import NoisyCurrentSource
from pyNN.utility.plotting import Figure, Panel

def main():
    sim.setup(0.01)

    # Parameters of neuron model; these are the default parameters and are written for readability
    neuron_parameters = {
        "v_rest": -65.0,
        "cm": 1.0,
        "tau_m": 20.0,
        "tau_refrac": 0.1,
        "tau_syn_E": 5.0,
        "tau_syn_I": 5.0,
        "i_offset": 0.0,
        "v_reset": -65.0,
        "v_thresh": -50.0
    }

    runtime = 500

    neuron = sim.IF_curr_exp(**neuron_parameters)
    population = sim.Population(
        1000,
        neuron,
        structure=RandomStructure(boundary=Sphere(radius=150)),
        initial_values={"v": -70.0},
        label="IF_curr_exp"
    )

    time = np.arange(0.0, runtime, 1.0)
    amplitudes = 0.1 * np.sin(time * np.pi / 100.0)

    current = sim.StepCurrentSource(times=time.tolist(), amplitudes=amplitudes.tolist())
    population[:200].inject(current)

    population.record("v")

    sim.run(runtime)
    v_data = population.get_data().segments[0].filter(name="v")[0]

    Figure(
        Panel(v_data, ylabel="Membrane potential (mV)", data_labels=[population.label], yticks=True, xlim=(0, runtime))
    ).save("LIF.png")

    sim.end()

if __name__ == "__main__":
    main()