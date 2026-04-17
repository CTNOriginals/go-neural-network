package network

import (
	"fmt"
	"strings"

	"github.com/CTNOriginals/go-neural-network/layer"
)

type Network struct {
	Layers []*layer.Layer
}

func NewNetwork(defs []layer.Definition) *Network {
	var network = Network{
		Layers: make([]*layer.Layer, len(defs)),
	}

	for i, def := range defs {
		var lyr = layer.NewLayer(def, i)
		network.Layers[i] = lyr

		if i == 0 {
			continue
		}

		lyr.Connect(network.Layers[i-1])
	}

	return &network
}

func (this Network) String() string {
	var str strings.Builder

	for i, lyr := range this.Layers {
		fmt.Fprintf(&str, "---- layer %d ----\n%s\n", i, lyr.String())
	}

	return str.String()
}

func (this Network) StringState() string {
	var str strings.Builder

	for _, lyr := range this.Layers {
		fmt.Fprintf(&str, " L%d:\n", lyr.Index)

		for n, nrn := range lyr.Neurons {
			fmt.Fprintf(&str, "  N%d: %s", n, nrn.String())

			if n < len(lyr.Neurons) {
				str.WriteRune('\n')
			}
		}
	}

	return str.String()
}

func (this Network) InputLayer() *layer.Layer {
	return this.Layers[0]
}
func (this Network) OutputLayer() *layer.Layer {
	return this.Layers[len(this.Layers)-1]
}

func (this Network) Output() []float64 {
	return this.OutputLayer().Values()
}

func (this *Network) SetInputs(inputs []float64) {
	this.InputLayer().Set(inputs)
}

func (this *Network) Forward() {
	for _, lyr := range this.Layers[1:] {
		lyr.Forward()
	}
}

func (this Network) SetOutputDeltas(expected []float64) {
	if len(this.OutputLayer().Neurons) != len(expected) {
		panic(fmt.Sprintf(
			"ErrorValue requires expected count (%d) to be equal to neuron count (%d)",
			len(expected), len(this.OutputLayer().Neurons),
		))
	}

	var activator = this.OutputLayer().Definition.GetActivator()

	for i, nrn := range this.OutputLayer().Neurons {
		var diff = nrn.Value - expected[i]
		nrn.Delta = diff * activator.Backward(nrn.Raw)
	}
}

func (this Network) Backward(rate float64) {
	for i := len(this.Layers) - 1; i > 0; i-- {
		var lyr = this.Layers[i]
		lyr.Backward(rate)
	}
}

func (this Network) Test(inputs []float64) []float64 {
	this.SetInputs(inputs)
	this.Forward()
	return this.Output()
}
