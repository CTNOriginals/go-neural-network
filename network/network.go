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

func (this *Network) Process() {
	for _, layer := range this.Layers[1:] {
		layer.Forward()
	}
}

func (this *Network) Test(inputs []float64) []float64 {
	this.SetInputs(inputs)
	this.Process()
	return this.Output()
}

func (this *Network) Train(inputs []float64, expect []float64, cycles int) {
	this.SetInputs(inputs)

	for cycle := range cycles {
		this.Process()

		var cost = this.OutputLayer().ErrorValue(expect)

		fmt.Printf("\n-- cycle %d --\n", cycle)
		fmt.Printf(
			"Inp %v\nOut: %v (%v)\nScore: %.2f\n",
			inputs,
			this.Output(),
			expect,
			cost,
		)

		this.OutputLayer().Backward(cost)
	}
}
