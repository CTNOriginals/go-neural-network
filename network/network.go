package network

import (
	"fmt"
	"strings"
)

type Network struct {
	Layers []*Layer
}

func NewNetwork(layers []LayerDefinition) *Network {
	var network = Network{
		Layers: make([]*Layer, len(layers)),
	}

	for i, def := range layers {
		var layer = NewLayer(def, i)
		network.Layers[i] = layer

		if i == 0 {
			continue
		}

		layer.Connect(network.Layers[i-1])
	}

	return &network
}

func (this Network) String() string {
	var str strings.Builder

	for i, layer := range this.Layers {
		fmt.Fprintf(&str, "---- layer %d ----\n%s\n", i, layer.String())
	}

	return str.String()
}

func (this Network) InputLayer() *Layer {
	return this.Layers[0]
}
func (this Network) OutputLayer() *Layer {
	return this.Layers[len(this.Layers)-1]
}

func (this Network) Output() []float64 {
	return this.OutputLayer().Values()
}

func (this *Network) SetInputs(inputs []float64) {
	this.InputLayer().Set(inputs)
}

func (this *Network) Forward() {
	for _, layer := range this.Layers[1:] {
		layer.Forward()
	}
}

func (this Network) SetOutputDeltas(expected []float64) {
	if len(this.OutputLayer().Neurons) != len(expected) {
		panic(fmt.Sprintf(
			"ErrorValue requires expected count (%d) to be equal to neuron count (%d)",
			len(expected), len(this.OutputLayer().Neurons),
		))
	}

	var activator = this.OutputLayer().definition.GetActivator()

	for i, neuron := range this.OutputLayer().Neurons {
		var diff = neuron.Value - expected[i]
		neuron.Delta = diff * activator.Backward(neuron.Raw)
	}
}

func (this Network) Backward(rate float64) {
	for i := len(this.Layers) - 1; i > 0; i-- {
		var layer = this.Layers[i]
		layer.Backward(rate)
	}
}

func (this Network) Test(inputs []float64) []float64 {
	this.SetInputs(inputs)
	this.Forward()
	return this.Output()
}
