package channn

import (
	"sync"
)


// MakeNeuronPerceptron creates a PERCEPTRON_TYPE neuron
// and calls ListenPerceptron() in a goroutine.
func MakePerceptronNeuron(bias float64) *Neuron {
	z := int32(0)
	n := &PerceptronNeuron{
		Neuron{
			InChan:     make(chan float64),
			Bias:       bias,
			NumIn:      &z,
			Control:    make(chan *ControlMessage),
			mutex:      &sync.Mutex{},
			OutWeights: make(map[*chan float64]float64),
			nType:       PERCEPTRON_TYPE,
		},
	}
	go n.Listen()
	return n
}
func (n PerceptronNeuron) GetType() NeuronType {
	return n.nType
}

type PerceptronNeuron struct {
	Neuron
}

// FirePerceptron sends 1.0 if the value is >= 1.0, otherwise sends 0.
func (n PerceptronNeuron) Fire(val float64) {
	for cp, w := range n.OutWeights {
		if (val >= 1.0) {
			*cp <- 1.0 * w
		} else {
			*cp <- 0
		}
	}
}
