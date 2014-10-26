package channn

import (
	"math/rand"
	"sync"
	"sync/atomic"
)

// MakeNeuronPerceptron sets all weights to the same value
func MakePerceptronNeuron(bias float64) *Neuron {
	z := int32(0)
	n := &Neuron{
		InChan:     make(chan float64),
		Bias:       bias,
		NumIn:      &z,
		Control:    make(chan *ControlMessage),
		mutex:      &sync.Mutex{},
		OutWeights: make(map[*chan float64]float64),
		nType:       PERCEPTRON_TYPE,
	}
	go n.Listen()
	return n
}

// Neuron represents a neuron in a network that is between either others.
// It receives inputs from a chan and calls Fire to send values to
// other Neurons or Outputs.
type Neuron struct {
	InChan chan float64
	NumIn  *int32
	Bias   float64

	mutex *sync.Mutex

	// Receives control messages in the Listen goroutine.
	Control chan *ControlMessage

	// This is a map of a pointer to an input channel
	// to the weight associated between Neurons.
	OutWeights map[*chan float64]float64

	nType      NeuronType
}

// AddInput accepts a pointer to an input chan for a node
// after.
func (n *Neuron) ConnectNeurons(next *Neuron) {
	// Add a pointer to the input and set a random weight
	n.OutWeights[&next.InChan] = rand.Float64()

	// Add an input counter for the next Neuron
	atomic.AddInt32(next.NumIn, 1)
}
func (n *Neuron) ConnectOutput(next *Output) {
	n.OutWeights[&next.InChan] = rand.Float64()
	next.Control <- &ControlMessage{
		Id: INCREMENT_INPUT,
	}
}

// Fire sends the result of the sigmoid function on the
// sum of (all input weights + bias).
func (n *Neuron) Fire(val float64) {
	for cp, w := range n.OutWeights {
		*cp <- Sigmoid(w * val)
	}
}

func (n *Neuron) ResetAllWeights(val float64) {
	// TODO: lock?
	n.mutex.Lock()
	for k, _ := range n.OutWeights {
		n.OutWeights[k] = val
	}
	n.mutex.Unlock()
}

// Listen reads all the inputs and calls the
// sigmoid function on the sum of all inputs
// and a bais weight.
func (n *Neuron) Listen() {
	n.mutex.Lock()
	var counter = *n.NumIn
	n.mutex.Unlock()
	var layerTotal float64
	for {
		select {
		case inVal := <-n.InChan:
			layerTotal += inVal
			counter--
			if counter == 0 {
				// The sum of the SUM(XiWj) + Bias
				n.Fire(layerTotal + n.Bias)
				layerTotal = 0
				counter = *n.NumIn
			}

		case ctlMsg := <-n.Control:
			switch ctlMsg.Id {
			case DESTROY:
				return
			case SET_WEIGHTS:
				n.ResetAllWeights(ctlMsg.Value.(float64))
			case SET_WEIGHT:
				key := ctlMsg.Key.(*chan float64)
				value := ctlMsg.Value.(float64)
				n.OutWeights[key] = value
			default:
				continue
			}
		}
	}
}
