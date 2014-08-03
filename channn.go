package channn

import (
	// "fmt"
	"math/rand"
	"sync"
	"sync/atomic"
)

// MakeNeuronPerceptron sets all weights to the same value
func MakeNeuronPerceptron(bias float64) *Neuron {
	z := int32(0)
	n := &Neuron{
		InChan:     make(chan float64),
		Bias:       bias,
		NumIn:      &z,
		Control:    make(chan *ControlMessage),
		mutex:      &sync.Mutex{},
		OutWeights: make(map[*chan float64]float64),
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

func (n *Neuron) Fire(val float64) {
	for cp, w := range n.OutWeights {
		*cp <- w * val
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
				result := layerTotal + n.Bias
				if result >= 1 {
					n.Fire(1)
				} else {
					n.Fire(0)
				}

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

func MakeInput() *Input {
	return &Input{
		Control:    make(chan *ControlMessage),
		OutWeights: make(map[*chan float64]float64),
		mutex:      &sync.Mutex{},
	}
}

type Input struct {
	OutWeights map[*chan float64]float64
	Control    chan *ControlMessage
	mutex      *sync.Mutex
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

func MakeOutput(bias float64) *Output {
	z := int32(0)
	o := &Output{
		InChan:  make(chan float64),
		NumIn:   &z,
		Control: make(chan *ControlMessage),
		Bias:    bias,
		OutChan: make(chan float64),
		mutex:   &sync.Mutex{},
	}
	go o.Listen()
	return o
}

type Output struct {
	InChan  chan float64
	NumIn   *int32
	Bias    float64
	Control chan *ControlMessage
	Result  float64
	OutChan chan float64
	mutex   *sync.Mutex
}

func (o *Output) Fire(val float64) {
	if val >= 1.0 {
		o.OutChan <- 1.0
	} else {
		o.OutChan <- 0.0
	}
}

func (o *Output) GetResult() float64 {
	return <-o.OutChan
}

func (o *Output) Listen() {
	var counter = *o.NumIn
	var layerTotal float64
	for {
		select {
		case inVal := <-o.InChan:
			layerTotal += inVal
			counter--

			if counter == 0 {
				// The sum of the SUM(XiWj) + Bias
				result := layerTotal + o.Bias
				if result >= 1 {
					o.Fire(1)
				} else {
					o.Fire(0)
				}

				layerTotal = 0.0
				counter = *o.NumIn
			}

		case ctlMsg := <-o.Control:
			switch ctlMsg.Id {
			case DESTROY:
				return
			case INCREMENT_INPUT:
				cur := (*o.NumIn + 1)
				o.NumIn = &cur
				counter = *o.NumIn

			default:
				continue
			}
		}
	}
}
