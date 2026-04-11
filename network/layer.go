package network

import "github.com/CTNOriginals/go-neural-network/formulas"

type Layer struct {
	Neurons         []*Neuron
	ActivatorType   formulas.TActivator
	InitializerType formulas.TInitializer
}

func NewLayer(size int, activatorType formulas.TActivator, initializerType formulas.TInitializer) *Layer {
	var neurons = make([]*Neuron, size)

	var activator = formulas.Activators[activatorType]
	var initializer = formulas.Initializers[initializerType]

	for i := range len(neurons) {
		neurons[i] = &Neuron{
			Weights: make([]*Connection, 0),
			Bias:    initializer(),
			Value:   0,

			activator: activator,
		}
	}

	return &Layer{
		Neurons:         neurons,
		ActivatorType:   activatorType,
		InitializerType: initializerType,
	}
}

// Connects each neuron in this to
// each neuron in the previous layer
// and assigns a weight to that connection.
func (this Layer) Connect(source *Layer) {
	var initializer = formulas.Initializers[this.InitializerType]

	for _, neuron := range this.Neurons {
		neuron.Weights = make([]*Connection, len(source.Neurons))

		for i, origin := range source.Neurons {
			neuron.Weights[i] = &Connection{
				Origin: origin,
				Weight: initializer(),
			}
		}
	}
}
