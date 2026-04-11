package formulas

import (
	"fmt"
	"math"
)

type TActivator int

const (
	ReLU TActivator = iota
	LeakyReLU
	Sigmoid
)

var ActivatorNames = []string{
	"ReLU",
	"LeakyReLU",
	"Sigmoid",
}

type TActivatorFn func(x float64) float64

type Activator struct {
	Forward  TActivatorFn
	Backward TActivatorFn
}

var Activators = map[TActivator]Activator{
	ReLU: {
		Forward: func(x float64) float64 {
			return max(0, x)
		},
		Backward: func(x float64) float64 {
			if x < 0 {
				return 0
			}

			return 1
		},
	},
	LeakyReLU: {
		Forward: func(x float64) float64 {
			if x >= 0 {
				return x
			}

			return x * 0.01
		},
		Backward: func(x float64) float64 {
			if x >= 0 {
				return 1
			}

			return 0.01
		},
	},
	Sigmoid: {
		Forward: func(x float64) float64 {
			return 1 / (1 + math.Pow(math.E, -x))
		},
		Backward: func(x float64) float64 {
			var sigmoid = 1 / (1 + math.Pow(math.E, -x))
			return sigmoid * (1 - sigmoid)
		},
	},
}

func PrintResults(x float64) {
	fmt.Printf("--- %.2f ----\n", x)

	for i, name := range ActivatorNames {
		var fn = Activators[TActivator(i)]
		fmt.Printf("%s:   \t %.2f\t%.2f\n", name, fn.Forward(x), fn.Backward(x))
	}
}
