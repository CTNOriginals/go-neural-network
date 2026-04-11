package network

type Layer []*Neuron

func NewLayer(size int) *Layer {
	return &Layer{}
}

// Connects each neuron in this to
// each neuron in the previous layer
// and assigns a weight to that connection.
func (this Layer) Connect(layer *Layer) {

}
