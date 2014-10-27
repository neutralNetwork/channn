package channn

import (
	"sync"
)


// MakeSigmoidNeuron creates a SIGMOID_TYPE neuron and calls
// Listen() in a goroutine.
func MakeSigmoidNeuron(bias float64) *Neuron {
	z := int32(0)
	n := &SigmoidNeuron{
		Neuron{
			InChan:     make(chan float64),
			Bias:       bias,
			NumIn:      &z,
			Control:    make(chan *ControlMessage),
			mutex:      &sync.Mutex{},
			OutWeights: make(map[*chan float64]float64),
			nType:       SIGMOID_TYPE,
		},}
	go n.Listen()
	return n
}

type SigmoidNeuron struct {
	Neuron
}

// Fire sends the result of the sigmoid function on the
// sum of (all input weights + bias).
func (sn SigmoidNeuron) Fire(val float64) {
	for nextPointer, w := range sn.OutWeights {
		*nextPointer <- Sigmoid(w * val)
	}
}
