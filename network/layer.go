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
	index      int
}

func NewLayer(def LayerDefinition, index int) *Layer {
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
		index:       index,
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

func (this Layer) StringCompact() string {
	return fmt.Sprintf(
		"Layer ID:%d S:%d I:%s A:%s",
		this.index,
		this.definition.Size,
		this.definition.InitializerType.String(),
		this.definition.ActivatorType.String(),
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

func (this Layer) ErrorValue(expected []float64) float64 {
	if len(this.Neurons) != len(expected) {
		this.error(
			"ErrorValue requires expected count (%d) to be equal to neuron count (%d)",
			len(expected), len(this.Neurons),
		)
	}

	return 0
}

func (this Layer) error(format string, args ...any) {
	var content = fmt.Sprintf(format, args...)
	var msg = fmt.Sprintf(
		"%s\n%s",
		this.StringCompact(),
		content,
	)

	panic(msg)
}
