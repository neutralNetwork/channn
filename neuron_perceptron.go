package channn

import (
	// "fmt"
	"sync"
)


// MakeNeuronPerceptron creates a PERCEPTRON_TYPE neuron
// and calls ListenPerceptron() in a goroutine.
func MakePerceptronNeuron(bias float64) *PerceptronNeuron {
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

type PerceptronNeuron struct {
	Neuron
}

// Fire sends 1.0 if the value is >= 1.0, otherwise sends 0.
func (n *PerceptronNeuron) Fire(val float64) {
	for cp, w := range n.OutWeights {
		if (val >= 1.0) {
			*cp <- 1.0 * w
		} else {
			*cp <- 0
		}
	}
}

// Listen reads all the inputs and calls the Fire method
// when all values of the input have been received.
func (n *PerceptronNeuron) Listen() {
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
