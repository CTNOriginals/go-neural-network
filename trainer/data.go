package trainer

import (
	"math"
	"math/rand"
)

type TrainingData []Sample

func (this TrainingData) Inputs() [][]float64 {
	var inputs = make([][]float64, len(this))

	for i, sample := range this {
		inputs[i] = sample.Inputs
	}

	return inputs
}

func (this *TrainingData) Push(samples ...Sample) {
	*this = append(*this, samples...)
}

func (this TrainingData) GetRandonSample() Sample {
	var rng = rand.Float64()
	var index = math.Round(rng * float64(len(this)-1))

	return this[int(index)]
}
