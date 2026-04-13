package network

import (
	"fmt"
	"strings"

	"github.com/CTNOriginals/go-neural-network/formulas"
)

type LayerDefinition struct {
	Size            int
	InitializerType formulas.TInitializer
	ActivatorType   formulas.TActivator
}

type Layer struct {
	Neurons     []*Neuron
	Initializer formulas.TInitializerFn
	Activator   formulas.Activator

	definition LayerDefinition
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
		Initializer: initializer,
		Activator:   activator,
		definition:  def,
	}
}

func (this Layer) String() string {
	var neurons strings.Builder

	for i, neuron := range this.Neurons {
		fmt.Fprintf(&neurons,
			" %d: %s",
			i,
			neuron.String(),
		)

		if i < len(this.Neurons)-1 {
			neurons.WriteRune('\n')
		}
	}

	return fmt.Sprintf(
		"Size: %d\nInitializer: %s\nActivation: %s\n%s\n",
		this.definition.Size,
		this.definition.InitializerType.String(),
		this.definition.ActivatorType.String(),
		neurons.String(),
	)
}

// Connects each neuron in this to
// each neuron in the previous layer
// and assigns a weight to that connection.
func (this *Layer) Connect(source *Layer) {
	var src = *source

	for _, neuron := range this.Neurons {
		for _, origin := range src.Neurons {
			var connection = &Connection{
				Origin: origin,
				Weight: this.Initializer(),
			}

			neuron.Weights = append(neuron.Weights, connection)
		}
	}
}
