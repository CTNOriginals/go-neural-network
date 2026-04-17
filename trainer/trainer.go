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
	for cycle := range cycles {
		var sample = this.Data.GetRandonSample()

		if (cycle+1)%(cycles/3) == 0 {
			fmt.Printf("---- Training Cycle %d ----\n", cycle)
			fmt.Printf("%v", this.Network.String())
		}

		this.Network.SetInputs(sample.Inputs)
		this.Network.Forward()
		this.Network.SetOutputDeltas(sample.Expect)
		this.Network.Backward(rate)
	}
}
