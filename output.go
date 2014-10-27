package channn

import (
	"fmt"
	"sync"
)

type ChannnOutput interface {
	GetResult() float64
}

// Output
type Output struct {
	InChan  chan float64
	NumIn   *int32
	Bias    float64

	mutex   *sync.Mutex

	Control chan *ControlMessage

	Result  float64

	OutChan chan float64

	nType NeuronType

}
func (o Output) GetType() NeuronType {
	return o.nType
}

func (o *Output) String() string {
	return fmt.Sprintf("output %s", &o)
}

func (o Output) GetResult() float64 {
	o.Result = <-o.OutChan
	return o.Result
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

		case ctlMsg := <-o.Control:
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
