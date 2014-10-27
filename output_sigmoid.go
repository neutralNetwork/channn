package channn

import (
	"sync"
)

func MakeSigmoidOutput(bias float64) *Output {
	z := int32(0)
	o := &SigmoidOutput{
		InChan:  make(chan float64),
		NumIn:   &z,
		Control: make(chan *ControlMessage),
		Bias:    bias,
		OutChan: make(chan float64),
		mutex:   &sync.Mutex{},
		nType:   SIGMOID_TYPE,
	}
	go o.ListenSigmoid()
	return o
}


type SigmoidOutput struct {
	Output
	nType NeuronType
}


// FireSigmoid accepts the value which is the sum of all inputs
// with the bias added; it sends the result of calling the sigmoid
// function on this value.
func (so *SigmoidOutput) Fire(val float64) {
	so.OutChan <- Sigmoid(val)
}
