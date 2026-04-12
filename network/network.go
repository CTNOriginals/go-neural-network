package network

type Network struct {
	Layers []*Layer
}

func NewNetwork(layers []LayerDefinition) *Network {
	var network = Network{
		Layers: make([]*Layer, len(layers)),
	}

	for i, def := range layers {
		var layer = NewLayer(def)

		if i == 0 {
			continue
		}

		layer.Connect(network.Layers[i-1])

		network.Layers[i] = layer
	}

	return &network
}
