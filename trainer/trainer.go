package trainer

import (
	"fmt"
	"math"
	"strings"

	"github.com/CTNOriginals/go-neural-network/network"
)

type Trainer struct {
	Network *network.Network
	Data    *TrainingData
}

func NewTrainer(net *network.Network) *Trainer {
	var data = make(TrainingData, 0)

	return &Trainer{
		Network: net,
		Data:    &data,
	}
}

func (this Trainer) Train(rate float64, cycles int) {
	var qsize = 3
	var origState = make([]string, qsize)

	for cycle := range cycles {
		var sample = this.Data.GetRandonSample()
		// var sample = (*this.Data)[cycle%len(*this.Data)]
		// var sample = (*this.Data)[len(*this.Data)/2]

		origState = append(origState[1:qsize], this.Network.StringState())

		this.Network.SetInputs(sample.Inputs)
		this.Network.Forward()
		this.Network.SetOutputDeltas(sample.Expect)
		this.Network.Backward(rate)

		for _, lyr := range this.Network.Layers {
			for _, nrn := range lyr.Neurons {
				if math.Abs(nrn.Value) > 10 {
					fmt.Println("---- EXPLODING ----")
					this.logCycle(cycle, strings.Join(origState, "\n"))
					return
				}
			}
		}

		// if (cycle+1)%max(max(cycles, 1)/3, 1) == 0 {
		// 	this.logCycle(cycle, strings.Join(origState, "\n"))
		// }
	}
}

func (this Trainer) logCycle(cycle int, origState string) {
	fmt.Printf("---- Training Cycle %d ----\n", cycle)
	fmt.Printf("-- History --\n%v", origState)
	fmt.Printf("-- Current --\n%v", this.Network.StringState())
}
