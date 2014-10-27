package channn

import (
	// "fmt"
	"math/rand"
	"sync"
)

type Input struct {
	OutWeights map[*chan float64]float64
	Control    chan *ControlMessage
	mutex      *sync.Mutex
	nType      NeuronType
}

func (i Input) GetType() NeuronType {
	return i.nType
}
// addOutput adds a pointer to the input and set a random weight.
func (i *Input) addOutput(c *chan float64) {
	i.mutex.Lock()
	i.OutWeights[c] = rand.Float64()
	i.mutex.Unlock()

}
func (i *Input) ConnectNeurons(next *ChanNeuron) {
	ntype := (*next).GetType()
	msg := &ControlMessage{
		Id: INCREMENT_INPUT,
	}

	if ntype == SIGMOID_TYPE {
		n, success := (*next).(SigmoidNeuron); if !success {
			panic("failed to case sigmoid neuron")
		}
		i.addOutput(&n.InChan)
		n.Control <- msg

	} else if ntype == PERCEPTRON_TYPE {
		n, success := (*next).(PerceptronNeuron); if !success {
			panic("failed to case perceptron neuron")
		}
		i.addOutput(&n.InChan)
		n.Control <- msg
	}

	// TODO: implement output types for perceptron and sigmoid types
	// } else if ntype == OUTPUT_TYPE {
	// 	n, success := (*next).(); if !success {
	// 		panic("failed to case perceptron neuron")
	// 	}
	// 	ptr = &n.InChan
	// 	n.Control <- msg
	// }



}

// Fire sends the input value to the next layer of neurons after
// multiplying it by the corresponding weight.
func (i Input) Fire(val float64) {
	for nextChannel, weight := range i.OutWeights {
		*nextChannel <- val * weight
	}
}

func (i *Input) ResetAllWeights(val float64) {
	i.mutex.Lock()
	for k, _ := range i.OutWeights {
		i.OutWeights[k] = val
	}
	i.mutex.Unlock()
}
