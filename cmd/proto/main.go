package main

import (
	"fmt"
	"time"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/network"
)

func main() {
	var startTime = time.Now()
	fmt.Printf("\n\n---- go-neural-network START %s ----\n", startTime.Format(time.TimeOnly))
	defer func() {
		fmt.Printf("\n---- go-neural-network END %s (%f) ----\n", startTime.Format(time.TimeOnly), time.Since(startTime).Seconds())
	}()

	var notGate = []network.LayerDefinition{
		{Size: 1},
		{
			Size: 1,
			Initializers: network.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.LeakyReLU,
		},
		{
			Size: 1,
			Initializers: network.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	}
	var xorGate = []network.LayerDefinition{
		{Size: 2},
		{Size: 3,
			Initializers: network.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.LeakyReLU,
		},
		{Size: 1,
			Initializers: network.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	}

	_ = notGate
	_ = xorGate

	var nn = network.NewNetwork(xorGate)
	// nn.Layers[len(nn.Layers)-1].Neurons[0].Weights[0].Weight = 0
	// nn.Layers[len(nn.Layers)-1].Neurons[0].Bias = -2

	fmt.Print(nn.String())
	nn.Train(
		[]float64{1, 0},
		[]float64{1},
		0.1,
		1000,
	)
	fmt.Print(nn.String())
	nn.Train(
		[]float64{0, 0},
		[]float64{0},
		0.1,
		1000,
	)
	fmt.Print(nn.String())
	nn.Train(
		[]float64{1, 1},
		[]float64{0},
		0.1,
		1000,
	)
	fmt.Print(nn.String())
	fmt.Print(nn.String())
	nn.Train(
		[]float64{0, 1},
		[]float64{1},
		0.1,
		1000,
	)
	fmt.Print(nn.String())

	fmt.Print(nn.Test([]float64{1, 1}))
}
