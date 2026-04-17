package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

func RunXorGateDemo() {
	fmt.Println("========================================")
	fmt.Println("       XOR Gate Learning Demo")
	fmt.Println("========================================")

	defs := []layer.Definition{
		{
			Size: 2,
		},
		{
			Size:          4,
			Initializers: layer.InitializerTypes{Weight: formulas.Random, Bias: formulas.Zero},
			ActivatorType: formulas.LeakyReLU,
		},
		{
			Size:          1,
			Initializers: layer.InitializerTypes{Weight: formulas.Random, Bias: formulas.Zero},
			ActivatorType: formulas.Sigmoid,
		},
	}

	net := network.NewNetwork(defs)

	data := trainer.TrainingData{
		{Inputs: []float64{0, 0}, Expect: []float64{0}},
		{Inputs: []float64{1, 0}, Expect: []float64{1}},
		{Inputs: []float64{0, 1}, Expect: []float64{1}},
		{Inputs: []float64{1, 1}, Expect: []float64{0}},
	}

	tr := trainer.NewTrainer(net)
	*tr.Data = data

	learningRate := 0.1
	cycles := 50000

	fmt.Printf("\n--- Starting Training ---\n")
	fmt.Printf("Learning Rate: %.1f\n", learningRate)
	fmt.Printf("Cycles: %d\n", cycles)
	fmt.Printf("Training Samples: %d\n", len(data))

	fmt.Printf("\nTraining...\n")

	var printInterval = cycles / 5

	for cycle := 0; cycle < cycles; cycle++ {
		sample := tr.Data.GetRandonSample()

		net.SetInputs(sample.Inputs)
		net.Forward()
		net.SetOutputDeltas(sample.Expect)
		net.Backward(learningRate)

		if (cycle+1)%printInterval == 0 {
			var totalError float64
			for _, s := range data {
				net.SetInputs(s.Inputs)
				net.Forward()
				output := net.Output()[0]
				totalError += math.Abs(output - s.Expect[0])
			}
			accuracy := 1 - (totalError / float64(len(data)))
			fmt.Printf("Cycle %d: Accuracy: %.2f%%\n", cycle+1, accuracy*100)
		}
	}

	fmt.Println("\n--- Training Complete ---")

	testCases := []struct {
		inputs []float64
		expect float64
	}{
		{inputs: []float64{0, 0}, expect: 0},
		{inputs: []float64{1, 0}, expect: 1},
		{inputs: []float64{0, 1}, expect: 1},
		{inputs: []float64{1, 1}, expect: 0},
	}

	fmt.Println("\n========================================")
	fmt.Println("          Test Results")
	fmt.Println("========================================")
	fmt.Printf("\n%-10s | %-10s | %-10s | %-8s\n", "Input 1", "Input 2", "Expected", "Output")
	fmt.Println(strings.Repeat("-", 46))

	var totalError float64

	for _, tc := range testCases {
		output := net.Test(tc.inputs)[0]
		rounded := math.Round(output)
		diff := math.Abs(output - tc.expect)
		totalError += diff

		fmt.Printf("%-10.1f | %-10.1f | %-10.1f | %-8.2f", tc.inputs[0], tc.inputs[1], tc.expect, output)
		if math.Abs(rounded-tc.expect) < 0.5 {
			fmt.Printf(" [OK]")
		} else {
			fmt.Printf(" [FAIL]")
		}
		fmt.Println()
	}

	avgError := totalError / float64(len(testCases))
	accuracy := (1 - avgError) * 100

	fmt.Println(strings.Repeat("-", 46))
	fmt.Printf("\nFinal Accuracy: %.2f%%\n", accuracy)
	fmt.Printf("Average Error: %.4f\n", avgError)

	fmt.Println("\n========================================")
}