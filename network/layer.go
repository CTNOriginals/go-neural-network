package network

import "github.com/CTNOriginals/go-neural-network/formulas"

type LayerDefinition struct {
	Size            int
	ActivatorType   formulas.TActivator
	InitializerType formulas.TInitializer
}

type Layer struct {
	Neurons     []*Neuron
	Activator   formulas.Activator
	Initializer formulas.TInitializerFn
}

func NewLayer(def LayerDefinition) *Layer {
	var neurons = make([]*Neuron, def.Size)

	var initializer = formulas.Initializers[def.InitializerType]
	var activator = formulas.Activators[def.ActivatorType]

	for i := range len(neurons) {
		neurons[i] = &Neuron{
			Weights: make([]*Connection, 0),
			Bias:    initializer(),
			Value:   0,

			activator: activator,
		}
	}

	return &Layer{
		Neurons:     neurons,
		Activator:   activator,
		Initializer: initializer,
	}
}

// Connects each neuron in this to
// each neuron in the previous layer
// and assigns a weight to that connection.
func (this Layer) Connect(source *Layer) {

	for _, neuron := range this.Neurons {
		neuron.Weights = make([]*Connection, len(source.Neurons))

		for i, origin := range source.Neurons {
			neuron.Weights[i] = &Connection{
				Origin: origin,
				Weight: this.Initializer(),
			}
		}
	}
}
