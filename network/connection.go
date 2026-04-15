package network

import "fmt"

type Connection struct {
	Origin *Neuron
	Weight float64

	// The sum of values that were applied to Weight
	Gradient float64
}

func (this Connection) String() string {
	return fmt.Sprintf("%.2f", this.Weight)
}

func (this Connection) Value() float64 {
	return this.Origin.Value * this.Weight
}

func (this *Connection) Correct(rate, delta float64) {
	var change = rate * delta * this.Origin.Value
	this.Weight -= change
	this.Gradient += change
}
