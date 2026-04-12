package network

import "fmt"

type Network struct {
	Layers []*Layer
}

func NewNetwork(layers []LayerDefinition) *Network {
	var network = Network{
		Layers: make([]*Layer, len(layers)),
	}

	for i, def := range layers {
		var layer = NewLayer(def)
		network.Layers[i] = layer

		if i == 0 {
			continue
		}

		layer.Connect(network.Layers[i-1])
	}

	return &network
}

func (this Network) String() string {
	var layers = ""

	for i, layer := range this.Layers {
		layers += fmt.Sprintf("---- layer %d ----\n%s\n", i, layer.String())
	}

	return layers
}
