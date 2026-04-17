package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/CTNOriginals/go-neural-network/layer"
	"github.com/CTNOriginals/go-neural-network/network"
	"github.com/CTNOriginals/go-neural-network/trainer"
)

type NetworkDef struct {
	Layers  []layer.Definition
	Samples []trainer.Sample

	Network *network.Network
	Trainer *trainer.Trainer
}

func (this *NetworkDef) Generate() *NetworkDef {
	var net = network.NewNetwork(this.Layers)

	this.Network = net
	this.Trainer = trainer.NewTrainer(net)
	this.Trainer.Data.Push(this.Samples...)

	return this
}

func stringer(values []float64) string {
	var builder strings.Builder

	for _, value := range values {
		fmt.Fprintf(&builder, "%.2f  ", value)

		if value < 0 {
			builder.WriteRune(' ')
		}
	}

	return builder.String()
}

func main() {
	var startTime = time.Now()
	fmt.Printf("\n\n---- go-neural-network START %s ----\n", startTime.Format(time.TimeOnly))
	defer func() {
		fmt.Printf("\n---- go-neural-network END %s (%f) ----\n", startTime.Format(time.TimeOnly), time.Since(startTime).Seconds())
	}()

	var args = os.Args[1:]
	var rate = 0.5
	var cycles = 3 //000000

	if len(args) > 1 {
		cycles, _ = strconv.Atoi(args[1])
	} else if len(args) > 0 {
		rate, _ = strconv.ParseFloat(args[0], 64)
	}

	var nn = NotGate.Generate()
	nn.Trainer.Train(rate, cycles)
	// fmt.Print(nn.Network.String())

	// for _, sample := range *nn.Trainer.Data {
	// 	fmt.Printf("Inputs: %s\n", stringer(sample.Inputs))
	// 	fmt.Printf("Output: %s\n\n", stringer(nn.Network.Test(sample.Inputs)))
	// }
}
