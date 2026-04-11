package network

type Connection struct {
	Origin *Neuron
	Weight float64
}

func (this Connection) Value() float64 {
	return this.Origin.Value * this.Weight
}
