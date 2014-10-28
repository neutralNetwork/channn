package channn

import (
	// "fmt"
	"math/rand"
	"sync"
)

func MakeInput() *Input {
	return &Input{
		OutWeights: make(map[*chan float64]float64),
		Control:    make(chan *ControlMessage),
		mutex:      &sync.Mutex{},
		nType:      INPUT_TYPE,
	}
}

type Input struct {
	OutWeights map[*chan float64]float64
	Control    chan *ControlMessage
	mutex      *sync.Mutex
	nType      NeuronType
}

// func (i Input) GetType() NeuronType {
// 	return i.nType
// }
// addOutput adds a pointer to the input and set a random weight.
func (i *Input) addOutput(c *chan float64) {
	i.mutex.Lock()
	i.OutWeights[c] = rand.Float64()
	i.mutex.Unlock()
}

func (i *Input) ConnectNeurons(next ChanNeuron) {
	// Add weight and pointer to the next neuron's input.
	inChanPtr := next.GetInChanPtr()
	i.addOutput(inChanPtr)

	// Send message to increment the input.
	msg := &ControlMessage{
		Id: INCREMENT_INPUT,
	}
	next.ReceiveControlMsg(msg)
}

// In sends the input value to the next layer of neurons after
// multiplying it by the corresponding weight.
func (i *Input) In(val float64) {
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
