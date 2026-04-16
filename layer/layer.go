package layer

import (
	"fmt"
	"strings"

	"github.com/CTNOriginals/go-neural-network/formulas"
	"github.com/CTNOriginals/go-neural-network/neuron"
)

type InitializerTypes struct {
	Weight formulas.TInitializer
	Bias   formulas.TInitializer
}

type LayerDefinition struct {
	Size          int
	Initializers  InitializerTypes
	ActivatorType formulas.TActivator
}

func (this LayerDefinition) GetActivator() formulas.Activator {
	return formulas.Activators[this.ActivatorType]
}
func (this LayerDefinition) GetWeightInit() formulas.TInitializerFn {
	return formulas.Initializers[this.Initializers.Weight]
}
func (this LayerDefinition) GetBiasInit() formulas.TInitializerFn {
	return formulas.Initializers[this.Initializers.Bias]
}

type Layer struct {
	Neurons []*neuron.Neuron

	definition LayerDefinition
	index      int
}

func NewLayer(def LayerDefinition, index int) *Layer {
	var nrns = make([]*neuron.Neuron, def.Size)

	var activator = def.GetActivator()

	for i := range len(nrns) {
		nrns[i] = neuron.NewNeuron(def.GetBiasInit()(), activator)
	}

	return &Layer{
		Neurons:    nrns,
		definition: def,
		index:      index,
	}
}

func (this Layer) GetDefinition() LayerDefinition {
	return this.definition
}

func (this Layer) String() string {
	var nrns strings.Builder

	for i, nrn := range this.Neurons {
		fmt.Fprintf(&nrns,
			" %d: %s",
			i,
			nrn.String(),
		)

		if i < len(this.Neurons)-1 {
			nrns.WriteRune('\n')
		}
	}

	return fmt.Sprintf(
		"Size: %d\nInitializer: W-%s B-%s\nActivation: %s\n%s\n",
		this.definition.Size,
		this.definition.Initializers.Weight.String(),
		this.definition.Initializers.Bias.String(),
		this.definition.ActivatorType.String(),
		nrns.String(),
	)
}

func (this Layer) StringCompact() string {
	return fmt.Sprintf(
		"Layer ID:%d S:%d IW:%s IB:%s A:%s",
		this.index,
		this.definition.Size,
		this.definition.Initializers.Weight.String(),
		this.definition.Initializers.Bias.String(),
		this.definition.ActivatorType.String(),
	)
}

func (this *Layer) Connect(source *Layer) {
	var weightInit = this.definition.GetWeightInit()

	for _, nrn := range this.Neurons {
		for _, origin := range source.Neurons {
			var conn = neuron.NewConnection(
				origin, nrn,
				weightInit(),
			)

			nrn.Inputs = append(nrn.Inputs, conn)
			origin.Outputs = append(origin.Outputs, conn)
		}
	}
}

func (this Layer) Values() []float64 {
	var values = make([]float64, len(this.Neurons))

	for i, nrn := range this.Neurons {
		values[i] = nrn.Value
	}

	return values
}

func (this *Layer) Set(values []float64) {
	if len(values) != len(this.Neurons) {
		this.error(
			"ErrorValue requires value count (%d) to be equal to neuron count (%d)",
			len(values), len(this.Neurons),
		)
	}

	for i, nrn := range this.Neurons {
		nrn.Value = values[i]
	}
}

func (this *Layer) Forward() {
	for _, nrn := range this.Neurons {
		nrn.Forward()
	}
}

func (this *Layer) Backward(rate float64) {
	for _, nrn := range this.Neurons {
		nrn.Backward(rate)
	}
}

func (this Layer) error(format string, args ...any) {
	var content = fmt.Sprintf(format, args...)
	var msg = fmt.Sprintf(
		"%s\n%s",
		this.StringCompact(),
		content,
	)

	panic(msg)
}
