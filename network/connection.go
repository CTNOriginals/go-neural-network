package network

import "fmt"

type Connection struct {
	Source      *Neuron
	Destination *Neuron
	Weight      float64
	oldWeight   float64

	// The sum of values that were applied to Weight
	Gradient float64 // TODO:
}

func NewConnection(source *Neuron, dest *Neuron, weight float64) *Connection {
	return &Connection{
		Source:      source,
		Destination: dest,
		Weight:      weight,
		oldWeight:   weight,
	}
}

func (this Connection) String() string {
	return fmt.Sprintf("%.2f", this.Weight)
}

func (this Connection) Value() float64 {
	return this.Source.Value * this.Weight
}

func (this Connection) Delta() float64 {
	return this.Source.Value * this.oldWeight
}

func (this *Connection) Correct(rate, delta float64) {
	var change = rate * delta * this.Source.Value

	this.oldWeight = this.Weight
	this.Weight -= change
	this.Gradient += change
}
