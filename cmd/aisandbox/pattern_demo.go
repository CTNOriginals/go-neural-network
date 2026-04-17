package main

import (
	"fmt"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

func RunPatternDemo() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║           PATTERN LEARNING DEMO - 'Any Input is 1'          ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	inputDef := layer.Definition{
		Size:          3,
		Initializers:  layer.InitializerTypes{Weight: formulas.Random, Bias: formulas.Zero},
		ActivatorType: formulas.Sigmoid,
	}
	hiddenDef := layer.Definition{
		Size:          4,
		Initializers:  layer.InitializerTypes{Weight: formulas.Random, Bias: formulas.Zero},
		ActivatorType: formulas.Sigmoid,
	}
	outputDef := layer.Definition{
		Size:          1,
		Initializers:  layer.InitializerTypes{Weight: formulas.Random, Bias: formulas.Zero},
		ActivatorType: formulas.Sigmoid,
	}

	net := network.NewNetwork([]layer.Definition{inputDef, hiddenDef, outputDef})
	tr := trainer.NewTrainer(net)

	tr.Data.Push(
		trainer.Sample{Inputs: []float64{1, 0, 0}, Expect: []float64{1}},
		trainer.Sample{Inputs: []float64{0, 1, 0}, Expect: []float64{1}},
		trainer.Sample{Inputs: []float64{0, 0, 1}, Expect: []float64{1}},
		trainer.Sample{Inputs: []float64{0, 0, 0}, Expect: []float64{0}},
	)

	fmt.Println("Training on patterns:")
	fmt.Println("  [1,0,0] -> 1 (first input is 1)")
	fmt.Println("  [0,1,0] -> 1 (second input is 1)")
	fmt.Println("  [0,0,1] -> 1 (third input is 1)")
	fmt.Println("  [0,0,0] -> 0 (no inputs are 1)")
	fmt.Println()
	fmt.Println("Training network...")
	fmt.Printf("Learning rate: 0.1, Cycles: 50000\n")
	fmt.Println()

	tr.Train(0.1, 50000)

	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════════")
	fmt.Println("                        TEST RESULTS")
	fmt.Println("════════════════════════════════════════════════════════════════")
	fmt.Println()

	testPatterns := []struct {
		input []float64
		desc  string
	}{
		{[]float64{1, 0, 0}, "first input is 1"},
		{[]float64{0, 1, 0}, "second input is 1"},
		{[]float64{0, 0, 1}, "third input is 1"},
		{[]float64{1, 1, 0}, "first two inputs are 1"},
		{[]float64{1, 0, 1}, "first and third inputs are 1"},
		{[]float64{0, 1, 1}, "second and third inputs are 1"},
		{[]float64{1, 1, 1}, "all inputs are 1"},
		{[]float64{0, 0, 0}, "no inputs are 1"},
	}

	for _, tp := range testPatterns {
		output := net.Test(tp.input)
		result := output[0]
		predicted := "0"
		if result > 0.5 {
			predicted = "1"
		}
		expected := "0"
		hasOne := false
		for _, v := range tp.input {
			if v == 1 {
				hasOne = true
				break
			}
		}
		if hasOne {
			expected = "1"
		}

		fmt.Printf("Input: %v -> %s\n", tp.input, tp.desc)
		fmt.Printf("  Output: %.4f | Predicted: %s | Expected: %s", result, predicted, expected)
		if predicted == expected {
			fmt.Printf(" ✓\n")
		} else {
			fmt.Printf(" ✗\n")
		}
		fmt.Println()
	}
}