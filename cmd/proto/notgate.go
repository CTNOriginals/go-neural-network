package main

import (
	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

var NotGate = NetworkDef{
	Layers: []layer.Definition{
		{Size: 1},
		// {
		// 	Size: 1,
		// 	Initializers: layer.InitializerTypes{
		// 		Weight: formulas.Half,
		// 		Bias:   formulas.Zero,
		// 	},
		// 	ActivatorType: formulas.LeakyReLU,
		// },
		{
			Size: 1,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	},
	Samples: []trainer.Sample{{
		Inputs: []float64{1},
		Expect: []float64{0},
	}, {
		Inputs: []float64{0},
		Expect: []float64{1},
	}},
}
