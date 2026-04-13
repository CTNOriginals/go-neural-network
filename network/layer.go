package network

import (
	"fmt"
	"strings"

	"github.com/CTNOriginals/go-neural-network/formulas"
)

type InitializerTypes struct {
	Weight formulas.TInitializer
	Bias   formulas.TInitializer
}

type LayerDefinition struct {
	Size          int
	Initializers  InitializerTypes
	ActivatorType formulas.TActivator
}

type Layer struct {
	Neurons []*Neuron

	definition LayerDefinition
	index      int
}

func NewLayer(def LayerDefinition, index int) *Layer {
	var neurons = make([]*Neuron, def.Size)

	var biasInit = formulas.Initializers[def.Initializers.Bias]
	var activator = formulas.Activators[def.ActivatorType]

	for i := range len(neurons) {
		neurons[i] = &Neuron{
			Weights: make([]*Connection, 0),
			Bias:    biasInit(),
			Value:   0,

			activator: activator,
		}
	}

	return &Layer{
		Neurons:    neurons,
		definition: def,
		index:      index,
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
		"Size: %d\nInitializer: W-%s B-%s\nActivation: %s\n%s\n",
		this.definition.Size,
		this.definition.Initializers.Weight.String(),
		this.definition.Initializers.Bias.String(),
		this.definition.ActivatorType.String(),
		neurons.String(),
	)
}

func (this Layer) StringCompact() string {
	return fmt.Sprintf(
		"Layer ID:%d S:%d IW:%s IB:%s A:%s",
		this.index,
		this.definition.Size,
		this.definition.Initializers.Weight.String(),
		this.definition.Initializers.Bias.String(),
		this.definition.ActivatorType.String(),
	)
}

// Connects each neuron in this to
// each neuron in the previous layer
// and assigns a weight to that connection.
func (this *Layer) Connect(source *Layer) {
	var src = *source

	var weightInit = formulas.Initializers[this.definition.Initializers.Weight]

	for _, neuron := range this.Neurons {
		for _, origin := range src.Neurons {
			var connection = &Connection{
				Origin: origin,
				Weight: weightInit(),
			}

			neuron.Weights = append(neuron.Weights, connection)
		}
	}
}

func (this Layer) Values() []float64 {
	var values = make([]float64, len(this.Neurons))

	for i, neuron := range this.Neurons {
		values[i] = neuron.Value
	}

	return values
}

func (this *Layer) Set(values []float64) {
	if len(values) != len(this.Neurons) {
		this.error(
			"ErrorValue requires value count (%d) to be equal to neuron count (%d)",
			len(values), len(this.Neurons),
		)
	}

	for i, neuron := range this.Neurons {
		neuron.Value = values[i]
	}
}

func (this *Layer) Forward() {
	for _, neuron := range this.Neurons {
		neuron.Forward()
	}
}

func (this Layer) ErrorValue(expected []float64) float64 {
	if len(this.Neurons) != len(expected) {
		this.error(
			"ErrorValue requires expected count (%d) to be equal to neuron count (%d)",
			len(expected), len(this.Neurons),
		)
	}

	var cost float64 = 0

	for i, neuron := range this.Neurons {
		var diff = neuron.Value - expected[i]
		cost += diff * diff
	}

	cost = cost / float64(len(this.Neurons))

	return cost
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
