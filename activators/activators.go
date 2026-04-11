package activators

import "math"

type TMethod func(x float64) float64

type Activator struct {
	Forward  TMethod
	Backward TMethod
}

var ReLU = Activator{
	Forward: func(x float64) float64 {
		return max(0, x)
	},
	Backward: func(x float64) float64 {
		if x < 0 {
			return 0
		}

		return 1
	},
}

var leakySlope = 0.01
var LeakyReLU = Activator{
	Forward: func(x float64) float64 {
		if x >= 0 {
			return x
		}

		return x * leakySlope
	},
	Backward: func(x float64) float64 {
		if x >= 0 {
			return 1
		}

		return leakySlope
	},
}

var Sigmoid = Activator{
	Forward: func(x float64) float64 {
		return 1 / (1 + math.Pow(math.E, -x))
	},
	Backward: func(x float64) float64 {
		var sigmoid = 1 / (1 + math.Pow(math.E, -x))
		return sigmoid * (1 - sigmoid)
	},
}
