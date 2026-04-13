package network

import (
	"fmt"

	"github.com/CTNOriginals/go-neural-network/formulas"
)

type Neuron struct {
	Weights []*Connection
	Bias    float64
	Value   float64

	activator formulas.Activator
}

func (this Neuron) String() string {
	return fmt.Sprintf(
		"V%.4f B%.2f W%v",
		this.Value,
		this.Bias,
		this.Weights,
	)
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

func (this *Neuron) Forward() {
	this.Value = this.activator.Forward(this.Compute())
}

func (this *Neuron) Backward(cost float64) {
	for _, connection := range this.Weights {
		var correction = connection.Origin.Value
		correction *= cost
		correction *= this.activator.Backward(this.Value)

		connection.Weight *= correction
	}
}
