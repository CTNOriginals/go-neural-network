package trainer

import (
	"math"
	"math/rand"
)

type TrainingData []Sample

func (this *TrainingData) Push(samples ...Sample) {
	*this = append(*this, samples...)
}

func (this TrainingData) GetRandonSample() Sample {
	var rng = rand.Float64()
	var index = math.Round(rng * float64(len(this)-1))

	return this[int(index)]
}
