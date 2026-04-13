package formulas

import "math/rand"

type TInitializer int

const (
	Zero TInitializer = iota
	Half
	One
	Random
)

var InitializerNames = []string{
	"Zero",
	"Half",
	"One",
	"Random",
}

func (this TInitializer) String() string {
	return InitializerNames[this]
}

type TInitializerFn func() float64

var Initializers = map[TInitializer]TInitializerFn{
	Zero: func() float64 {
		return 0
	},
	Half: func() float64 {
		return 0.5
	},
	One: func() float64 {
		return 1
	},
	Random: func() float64 {
		return rand.Float64()
	},
}
