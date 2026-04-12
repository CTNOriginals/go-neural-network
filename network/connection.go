package network

import "fmt"

type Connection struct {
	Origin *Neuron
	Weight float64
}

func (this Connection) String() string {
	return fmt.Sprintf("%.2f", this.Weight)
}

func (this Connection) Value() float64 {
	return this.Origin.Value * this.Weight
}
