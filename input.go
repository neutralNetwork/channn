package channn

import (
	// "fmt"
	"math/rand"
	"sync"
	"sync/atomic"
)

func MakePerceptronInput() *Input {
	return &Input{
		Control:    make(chan *ControlMessage),
		OutWeights: make(map[*chan float64]float64),
		mutex:      &sync.Mutex{},
		nType:      PERCEPTRON_TYPE,
	}
}

type Input struct {
	OutWeights map[*chan float64]float64
	Control    chan *ControlMessage
	mutex      *sync.Mutex
	nType      NeuronType
}

func (i *Input) ConnectNeurons(next *Neuron) {
	// Add a pointer to the input and set a random weight
	i.OutWeights[&next.InChan] = rand.Float64()

	// Add an input counter for the next Neuron
	atomic.AddInt32(next.NumIn, 1)
}
func (i *Input) ConnectOutput(next *Output) {
	i.mutex.Lock()
	i.OutWeights[&next.InChan] = rand.Float64()
	i.mutex.Unlock()

	next.Control <- &ControlMessage{
		Id: INCREMENT_INPUT,
	}
}

// Fire sends the input value to the next layer of neurons after
// multiplying it by the corresponding weight.
func (i *Input) Fire(val float64) {
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
