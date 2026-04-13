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
