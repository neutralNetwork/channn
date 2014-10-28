package channn

import (
	"fmt"
	"sync"
	"math/rand"
)


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

	nType NeuronType
}
func (n *Neuron) String() string {
	return fmt.Sprintf("neuron %s", &n)
}

// Fire sends the result of the sigmoid function on the
// sum of (all input weights + bias).
func (sn *Neuron) Fire(val float64) {
	for nextPointer, w := range sn.OutWeights {
		*nextPointer <- Sigmoid(w * val)
	}
}

// addOutput adds a pointer to the input and set a random weight.
func (n *Neuron) addOutput(c *chan float64) {
	n.mutex.Lock()
	n.OutWeights[c] = rand.Float64()
	n.mutex.Unlock()

}
func (ne *Neuron) ConnectNeurons(next ChanNeuron) {
	// Add weight and pointer to the next neuron's input.
	inChanPtr := next.GetInChanPtr()
	ne.addOutput(inChanPtr)

	// Send message to increment the input.
	msg := &ControlMessage{
		Id: INCREMENT_INPUT,
	}
	next.ReceiveControlMsg(msg)
}

func (n *Neuron) ResetAllWeights(val float64) {
	n.mutex.Lock()
	for k, _ := range n.OutWeights {
		n.OutWeights[k] = val
	}
	n.mutex.Unlock()
}

//////// Satisfy the ChanNeuron interface.

// GetInChanPtr returns a pointer to the input channel
func (n *Neuron) GetInChanPtr() *chan float64 {
	return &n.InChan
}
func (n *Neuron) ReceiveControlMsg(msg *ControlMessage) {
	n.Control <- msg
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
				// layerTotal is the sum of the (Xi * Wj)
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
			case INCREMENT_INPUT:
				cur := (*n.NumIn + 1)
				n.NumIn = &cur
				counter = *n.NumIn
			default:
				continue
			}
		}
	}
}
