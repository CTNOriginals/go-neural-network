package main

import (
	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

var XorGate = NetworkDef{
	Layers: []layer.Definition{
		{Size: 2},
		{Size: 3,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Random,
			},
			ActivatorType: formulas.LeakyReLU,
		},
		{Size: 1,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	},
	Samples: []trainer.Sample{{
		Inputs: []float64{0, 0},
		Expect: []float64{0},
	}, {
		Inputs: []float64{1, 0},
		Expect: []float64{1},
	}, {
		Inputs: []float64{0, 1},
		Expect: []float64{1},
	}, {
		Inputs: []float64{1, 1},
		Expect: []float64{0},
	}},
}
