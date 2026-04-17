package main

import (
	"fmt"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

func RunLogicGatesDemo() {
	layers := []layer.Definition{
		{Size: 2},
		{
			Size: 3,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Random,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.LeakyReLU,
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

	net := network.NewNetwork(layers)
	t := trainer.NewTrainer(net)

	andData := []trainer.Sample{
		{Inputs: []float64{0, 0}, Expect: []float64{0}},
		{Inputs: []float64{0, 1}, Expect: []float64{0}},
		{Inputs: []float64{1, 0}, Expect: []float64{0}},
		{Inputs: []float64{1, 1}, Expect: []float64{1}},
	}

	orData := []trainer.Sample{
		{Inputs: []float64{0, 0}, Expect: []float64{0}},
		{Inputs: []float64{0, 1}, Expect: []float64{1}},
		{Inputs: []float64{1, 0}, Expect: []float64{1}},
		{Inputs: []float64{1, 1}, Expect: []float64{1}},
	}

	nandData := []trainer.Sample{
		{Inputs: []float64{0, 0}, Expect: []float64{1}},
		{Inputs: []float64{0, 1}, Expect: []float64{1}},
		{Inputs: []float64{1, 0}, Expect: []float64{1}},
		{Inputs: []float64{1, 1}, Expect: []float64{0}},
	}

	t.Data.Push(andData...)
	t.Data.Push(orData...)
	t.Data.Push(nandData...)

	fmt.Println("Training combined logic gates (AND, OR, NAND)...")
	t.Train(0.1, 50000)
	fmt.Println("Training complete!\n")

	testCases := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	fmt.Println("=== Test Results ===")
	fmt.Println("---------------------")

	results := make(map[string]float64)
	for _, inputs := range testCases {
		output := net.Test(inputs)[0]
		results[fmt.Sprintf("%.0f,%.0f", inputs[0], inputs[1])] = output
		fmt.Printf("Input: [%.0f, %.0f] => Output: %.4f\n", inputs[0], inputs[1], output)
	}

	fmt.Println("---------------------")

	andScore := calculateMatch(results, []float64{0, 0, 0, 1})
	orScore := calculateMatch(results, []float64{0, 1, 1, 1})
	nandScore := calculateMatch(results, []float64{1, 1, 1, 0})

	fmt.Println("\n=== Logic Gate Identification ===")
	fmt.Printf("AND:  %.2f%% match\n", andScore)
	fmt.Printf("OR:   %.2f%% match\n", orScore)
	fmt.Printf("NAND: %.2f%% match\n", nandScore)

	best := "AND"
	if orScore > andScore {
		best = "OR"
	}
	if nandScore > andScore && nandScore > orScore {
		best = "NAND"
	}
	fmt.Printf("\nBest match: %s\n", best)
}

func calculateMatch(results map[string]float64, expected []float64) float64 {
	inputs := []string{"0,0", "0,1", "1,0", "1,1"}
	matches := 0
	for i, key := range inputs {
		val := results[key]
		if (expected[i] == 1 && val >= 0.5) || (expected[i] == 0 && val < 0.5) {
			matches++
		}
	}
	return float64(matches) / 4.0 * 100
}