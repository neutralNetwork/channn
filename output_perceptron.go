package channn

import (
	"sync"
)


func MakePerceptronOutput(bias float64) *Output {
	z := int32(0)
	po := &Output{
		InChan:  make(chan float64),
		NumIn:   &z,
		Control: make(chan *ControlMessage),
		Bias:    bias,
		OutChan: make(chan float64),
		mutex:   &sync.Mutex{},
		nType:   PERCEPTRON_TYPE,
	}
	go po.ListenPerceptron()
	return po
}

type PerceptronOutput struct {
	Output
	nType NeuronType
}

// FirePerceptron conditionally sends a value of 1.0 or 0.0
// to the output channel.
func (po PerceptronOutput) Fire(val float64) {
	if val >= 1.0 {
		po.OutChan <- 1.0
	} else {
		po.OutChan <- 0.0
	}
}
