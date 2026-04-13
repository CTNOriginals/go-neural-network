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

	var layerDef = []network.LayerDefinition{
		{Size: 2},
		{
			Size: 3,
			Initializers: network.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.LeakyReLU,
		},
		{
			Size: 1,
			Initializers: network.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	}

	var nn = network.NewNetwork(layerDef)

	nn.Train(
		[]float64{1, 0},
		[]float64{1},
		10,
	)

	fmt.Print(nn.String())

	// nn.Layers[2].ErrorValue([]float64{1, 0})
	// var val = 0.64
	// fmt.Println(math.Sqrt(val))
	// fmt.Println(val * val)
}
