package activators

type TActivator func(val float64) float64

func ReLU(x float64) float64 {
	return max(0, x)
}

func LeakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}

	return x * 0.01
}
