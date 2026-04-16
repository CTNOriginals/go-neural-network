package trainer

type TrainingData []Sample

func (this *TrainingData) Push(samples ...Sample) {
	*this = append(*this, samples...)
}
