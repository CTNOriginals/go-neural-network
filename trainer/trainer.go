package trainer

import "github.com/CTNOriginals/go-neural-network/network"

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
