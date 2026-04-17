package trainer

import (
	"fmt"

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
	var origState string

	for cycle := range cycles {
		// var sample = this.Data.GetRandonSample()
		var sample = (*this.Data)[len(*this.Data)/2]

		origState = this.Network.StringState()

		this.Network.SetInputs(sample.Inputs)
		this.Network.Forward()
		this.Network.SetOutputDeltas(sample.Expect)
		this.Network.Backward(rate)

		if (cycle+1)%min(cycles, 1)/3 == 0 {
			fmt.Printf("---- Training Cycle %d ----\n", cycle)
			fmt.Printf("-- Original --\n%v", origState)
			fmt.Printf("-- Current  --\n%v", this.Network.StringState())
		}
	}
}
