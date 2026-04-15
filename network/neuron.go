package network

import (
	"fmt"

	"github.com/CTNOriginals/go-neural-network/formulas"
)

type Neuron struct {
	Inputs  []*Connection
	Outputs []*Connection

	Bias float64

	Raw   float64
	Value float64
	Delta float64

	activator formulas.Activator
}

func (this Neuron) String() string {
	return fmt.Sprintf(
		"V%.4f B%.2f W%v",
		this.Value,
		this.Bias,
		this.Inputs,
	)
}

// Compute calculates the raw combined value
// of this neuron and returns it.
func (this Neuron) Compute() float64 {
	var sum float64 = 0

	for _, connection := range this.Inputs {
		sum += connection.Value()
	}

	return sum + this.Bias
}

func (this *Neuron) Forward() {
	this.Raw = this.Compute()
	this.Value = this.activator.Forward(this.Raw)
}

func (this *Neuron) Backward(rate float64) {
	for _, connection := range this.Inputs {
		connection.Correct(rate, this.Delta)
	}

	this.Bias = rate * this.Delta
}
