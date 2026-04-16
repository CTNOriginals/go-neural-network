package main

import (
	"fmt"
	"time"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

func main() {
	var startTime = time.Now()
	fmt.Printf("\n\n---- go-neural-network START %s ----\n", startTime.Format(time.TimeOnly))
	defer func() {
		fmt.Printf("\n---- go-neural-network END %s (%f) ----\n", startTime.Format(time.TimeOnly), time.Since(startTime).Seconds())
	}()

	var notGate = []layer.Definition{
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
	}
	var xorGate = []layer.Definition{
		{Size: 2},
		{Size: 3,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.LeakyReLU,
		},
		{Size: 1,
			Initializers: layer.InitializerTypes{
				Weight: formulas.Half,
				Bias:   formulas.Zero,
			},
			ActivatorType: formulas.Sigmoid,
		},
	}

	_ = notGate
	_ = xorGate

	var nn = network.NewNetwork(notGate)
	var xorTrainer = trainer.NewTrainer(nn)
	var notTrainer = trainer.NewTrainer(nn)

	xorTrainer.Data.Push(
		trainer.Sample{
			Inputs: []float64{0, 0},
			Expect: []float64{0},
		},
		trainer.Sample{
			Inputs: []float64{1, 0},
			Expect: []float64{1},
		},
		trainer.Sample{
			Inputs: []float64{0, 1},
			Expect: []float64{1},
		},
		trainer.Sample{
			Inputs: []float64{1, 1},
			Expect: []float64{0},
		},
	)

	notTrainer.Data.Push(
		trainer.Sample{
			Inputs: []float64{1},
			Expect: []float64{0},
		},
		trainer.Sample{
			Inputs: []float64{0},
			Expect: []float64{1},
		},
	)

	// xorTrainer.Train(0.05, 100000)
	notTrainer.Train(0.05, 10000)

	fmt.Print(nn.String())
	fmt.Println(nn.Test([]float64{1}))
	fmt.Println(nn.Test([]float64{0}))
}
