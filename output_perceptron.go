package channn

import (
	// "fmt"
	"sync"
)


func MakePerceptronOutput(bias float64) *PerceptronOutput {
	z := int32(0)
	po := &PerceptronOutput{
		Output{
			InChan:  make(chan float64),
			NumIn:   &z,
			Control: make(chan *ControlMessage),
			Bias:    bias,
			OutChan: make(chan float64),
			mutex:   &sync.Mutex{},
			nType:   OUTPUT_TYPE,
		},
	}
	go po.Listen()
	return po
}

type PerceptronOutput struct {
	Output
}

// FirePerceptron conditionally sends a value of 1.0 or 0.0
// to the output channel.
func (po *PerceptronOutput) Fire(val float64) {
	if val >= 1.0 {
		po.OutChan <- 1.0
	} else {
		po.OutChan <- 0.0
	}
}

func (po *PerceptronOutput) Listen() {
	var counter = *po.NumIn
	var layerTotal float64
	for {
		select {
		case inVal := <-po.InChan:
			layerTotal += inVal
			counter--

			if counter == 0 {
				// The sum of the (Xi * Wj)
				po.Fire(layerTotal + po.Bias)
				layerTotal = 0.0
				counter = *po.NumIn
			}

		case ctlMsg := <-po.Control:
			switch ctlMsg.Id {
			case DESTROY:
				return
			case INCREMENT_INPUT:
				cur := (*po.NumIn + 1)
				po.NumIn = &cur
				counter = *po.NumIn

			default:
				continue
			}
		}
	}
}
