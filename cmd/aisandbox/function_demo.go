package main

import (
	"fmt"
	"math"

	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
	"github.com/CTNOriginals/go-neural-network/formulas"
)

func RunFunctionApproxDemo() {
	defs := []layer.Definition{
		{
			Size: 1,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
		{
			Size: 8,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
		{
			Size: 1,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	}

	net := network.NewNetwork(defs)
	tr := trainer.NewTrainer(net)

	sampleCount := 20
	for i := 0; i < sampleCount; i++ {
		x := float64(i) * math.Pi / float64(sampleCount-1)
		y := math.Sin(x)
		tr.Data.Push(trainer.Sample{Inputs: []float64{x}, Expect: []float64{y}})
	}

	rate := 0.1
	cycles := 80000

	fmt.Printf("Training sine function approximation...\n")
	fmt.Printf("Samples: %d, Cycles: %d, Rate: %.1f\n\n", sampleCount, cycles, rate)

	tr.Train(rate, cycles)

	fmt.Println("\n=== Testing Results ===")
	fmt.Println("X (radians) | Actual sin(x) | Predicted | Diff")
	fmt.Println("-----------|---------------|-----------|------")

	testPoints := []float64{0, math.Pi / 6, math.Pi / 4, math.Pi / 3, math.Pi / 2, 2 * math.Pi / 3, 3 * math.Pi / 4, 5 * math.Pi / 6, math.Pi}

	for _, x := range testPoints {
		actual := math.Sin(x)
		predicted := net.Test([]float64{x})[0]
		diff := predicted - actual
		fmt.Printf("  %.4f    |    %.4f      |   %.4f   | %.4f\n", x, actual, predicted, diff)
	}

	fmt.Println("\n=== Visualization ===")
	fmt.Println("Approximating sin(x) from 0 to PI:")

	for i := 0; i <= 20; i++ {
		x := float64(i) * math.Pi / 20
		actual := math.Sin(x)
		predicted := net.Test([]float64{x})[0]

		actualBar := int(actual * 20)
		predictedBar := int(predicted * 20)

		fmt.Printf("x=%.2f: ", x)
		for j := 0; j < actualBar; j++ {
			fmt.Print("*")
		}
		fmt.Println(" (actual)")

		fmt.Printf("        ")
		for j := 0; j < predictedBar; j++ {
			fmt.Print("#")
		}
		fmt.Println(" (predicted)")
	}
}