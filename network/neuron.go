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

// ComputeRaw calculates the raw combined value
// of this neuron and returns it.
func (this *Neuron) ComputeRaw() {
	var sum float64 = 0

	for _, connection := range this.Inputs {
		sum += connection.Value()
	}

	this.Raw = sum + this.Bias
}

func (this *Neuron) Forward() {
	this.ComputeRaw()
	this.Value = this.activator.Forward(this.Raw)
}

func (this *Neuron) ComputeDelta() {
	var sum float64 = 0

	for _, connection := range this.Outputs {
		sum += connection.Delta()
	}

	this.Delta = this.activator.Backward(this.Raw) * sum
}

func (this *Neuron) Backward(rate float64) {
	// Check if this neuron is an output neuron
	// if so, the delta should not be changed
	if len(this.Outputs) > 0 {
		this.ComputeDelta()
	}

	for _, connection := range this.Inputs {
		connection.Correct(rate, this.Delta)
	}

	this.Bias -= rate * this.Delta
}
