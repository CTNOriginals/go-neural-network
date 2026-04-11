package network

import "github.com/CTNOriginals/go-neural-network/activators"

type Neuron struct {
	Weights []Connection
	Bias    float64
	Value   float64

	activator activators.Activator
}

// Compute calculates the raw combined value
// of this neuron and returns it.
func (this Neuron) Compute() float64 {
	var sum float64 = 0

	for _, connection := range this.Weights {
		sum += connection.Value()
	}

	return sum + this.Bias
}

func (this *Neuron) Activate() {
	this.Value = this.activator.Forward(this.Compute())
}
