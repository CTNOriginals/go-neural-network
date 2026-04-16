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

	fmt.Print(nn.String())
	fmt.Print(nn.Test([]float64{1, 0}))
}
