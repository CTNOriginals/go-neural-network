package main

import (
	"fmt"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

func RunSimpleClassifierDemo() {
	defs := []layer.Definition{
		{
			Size: 2,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
		{
			Size: 4,
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

	sampleCount := 100
	for i := 0; i < sampleCount; i++ {
		for j := 0; j < sampleCount; j++ {
			x := float64(i) / float64(sampleCount-1)
			y := float64(j) / float64(sampleCount-1)

			var expected float64
			if y > x {
				expected = 1
			} else {
				expected = 0
			}

			tr.Data.Push(trainer.Sample{Inputs: []float64{x, y}, Expect: []float64{expected}})
		}
	}

	rate := 0.1
	cycles := 50000

	fmt.Println("Training classifier: points above/below diagonal line y = x...")
	fmt.Printf("Samples: %d, Cycles: %d, Rate: %.1f\n\n", sampleCount*sampleCount, cycles, rate)

	tr.Train(rate, cycles)

	fmt.Println("\n=== Classification Results ===")
	fmt.Println("Point (x,y)           | Expected | Predicted | Class")
	fmt.Println("----------------------|----------|-----------|------------------")

	testPoints := []struct {
		x, y     float64
		expected string
	}{
		{0.2, 0.8, "above"},
		{0.8, 0.2, "below"},
		{0.5, 0.5, "on line"},
		{0.3, 0.7, "above"},
		{0.7, 0.3, "below"},
		{0.4, 0.6, "above"},
		{0.6, 0.4, "below"},
		{0.1, 0.9, "above"},
		{0.9, 0.1, "below"},
		{0.45, 0.55, "above"},
	}

	classifications := make([]string, len(testPoints))

	for i, tp := range testPoints {
		predicted := net.Test([]float64{tp.x, tp.y})[0]
		if predicted > 0.5 {
			classifications[i] = "above"
		} else {
			classifications[i] = "below"
		}

		fmt.Printf("(%.1f, %.1f)             | %-7s |  %.4f   | %s\n",
			tp.x, tp.y, tp.expected, predicted, classifications[i])
	}

	fmt.Println("\n=== Decision Boundary Visualization ===")
	fmt.Println("Showing classification across a grid (y = x diagonal):")
	fmt.Println("  '*' = above line (class 1)")
	fmt.Println("  '.' = below line (class 0)")
	fmt.Println()

	gridSize := 10
	for i := gridSize; i >= 0; i-- {
		y := float64(i) / float64(gridSize)
		fmt.Printf("y=%.1f: ", y)

		for j := 0; j <= gridSize; j++ {
			x := float64(j) / float64(gridSize)
			predicted := net.Test([]float64{x, y})[0]

			if predicted > 0.5 {
				fmt.Print("*")
			} else {
				fmt.Print(".")
			}
		}
		fmt.Println()
	}

	fmt.Print("    ")
	for j := 0; j <= gridSize; j++ {
		x := float64(j) / float64(gridSize)
		fmt.Printf("x=%.1f ", x)
	}
	fmt.Println()

	fmt.Println("\n=== Summary ===")
	fmt.Println("Line equation: y = x")
	fmt.Println("Points above line (y > x): classified as class 1")
	fmt.Println("Points below line (y <= x): classified as class 0")

	correct := 0
	for i, tp := range testPoints {
		if classifications[i] == tp.expected {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(testPoints)) * 100
	fmt.Printf("Test accuracy: %.1f%% (%d/%d correct)\n", accuracy, correct, len(testPoints))

}

