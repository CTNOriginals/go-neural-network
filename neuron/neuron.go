package neuron

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

func NewNeuron(bias float64, activator formulas.Activator) *Neuron {
	return &Neuron{
		Inputs:    make([]*Connection, 0),
		Outputs:   make([]*Connection, 0),
		Bias:      bias,
		Value:     0,
		activator: activator,
	}
}

func (this Neuron) String() string {
	return fmt.Sprintf(
		"V%.4f B%.2f W%v D%.2f",
		this.Value,
		this.Bias,
		this.Inputs,
		this.Delta,
	)
}

func (this *Neuron) ComputeRaw() {
	var sum float64 = 0

	for _, conn := range this.Inputs {
		sum += conn.Value()
	}

	this.Raw = sum + this.Bias
}

func (this *Neuron) Forward() {
	this.ComputeRaw()
	this.Value = this.activator.Forward(this.Raw)
}

func (this *Neuron) ComputeDelta() {
	var sum float64 = 0

	for _, conn := range this.Outputs {
		sum += conn.Delta()
	}

	this.Delta = this.activator.Backward(this.Raw) * sum
}

func (this *Neuron) Backward(rate float64) {
	if len(this.Outputs) > 0 {
		this.ComputeDelta()
	}

	for _, conn := range this.Inputs {
		conn.Correct(rate, this.Delta)
	}

	this.Bias -= rate * this.Delta
}
