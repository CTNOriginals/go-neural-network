package layer

import "github.com/CTNOriginals/go-neural-network/formulas"

type InitializerTypes struct {
	Weight formulas.TInitializer
	Bias   formulas.TInitializer
}

type Definition struct {
	Size          int
	Initializers  InitializerTypes
	ActivatorType formulas.TActivator
}

func (this Definition) GetActivator() formulas.Activator {
	return formulas.Activators[this.ActivatorType]
}
func (this Definition) GetWeightInit() formulas.TInitializerFn {
	return formulas.Initializers[this.Initializers.Weight]
}
func (this Definition) GetBiasInit() formulas.TInitializerFn {
	return formulas.Initializers[this.Initializers.Bias]
}
