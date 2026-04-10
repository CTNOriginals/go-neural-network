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

		activation = fn(x) int
	}
*/
package neuron

type Connection struct {
	Origin *Neuron
	Weight int
}

func (this Connection) Value() int {
	return this.Origin.Value * this.Weight
}

type TActivatorFn func(val int) int

type Neuron struct {
	Weights []Connection
	Bias    int
	Value   int

	activator TActivatorFn
}

func (this Neuron) RawValue() int {
	var sum = 0

	for _, connection := range this.Weights {
		sum += connection.Value()
	}

	return sum + this.Bias
}

func (this Neuron) Activate() int {
	return this.activator(this.RawValue())
}
