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
func (n Neuron) GetType() NeuronType {
	return n.nType
}

func (n *Neuron) String() string {
	return fmt.Sprintf("neuron %s", &n)
}
// addOutput adds a pointer to the input and set a random weight.
func (n *Neuron) addOutput(c *chan float64) {
	n.mutex.Lock()
	n.OutWeights[c] = rand.Float64()
	n.mutex.Unlock()

}
func (ne *Neuron) ConnectNeurons(next *ChanNeuron) {
	ntype := (*next).GetType()
	msg := &ControlMessage{
		Id: INCREMENT_INPUT,
	}

	if ntype == SIGMOID_TYPE {
		n, success := (*next).(SigmoidNeuron); if !success {
			panic("failed to case sigmoid neuron")
		}
		n.addOutput(&n.InChan)
		n.Control <- msg

	} else if ntype == PERCEPTRON_TYPE {
		n, success := (*next).(PerceptronNeuron); if !success {
			panic("failed to case perceptron neuron")
		}
		ne.addOutput(&n.InChan)
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

func (n *Neuron) ResetAllWeights(val float64) {
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
			default:
				continue
			}
		}
	}
}
