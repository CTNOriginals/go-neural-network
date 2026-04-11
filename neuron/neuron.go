/*
Example neural net:
I  H  O
1  0  0
0  1  0

	H[0]: {
		I[0]: 1, W:0.5
		I[1]: 0, W:2

		x = (1*0.5)+(0*2)+bias
		x = 0.5

		activation = fn(x) float64
	}
*/
package neuron

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
	this.Value = this.activator(this.Compute())
}
