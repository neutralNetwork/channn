package channn

import (
	"fmt"
	"sync"
)

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

func (o *Output) String() string {
	return fmt.Sprintf("output %s", &o)
}

func (o Output) GetResult() float64 {
	o.Result = <-o.OutChan
	return o.Result
}

func (o *Output) GetInChanPtr() *chan float64 {
	return &o.InChan
}

func (po *Output) ReceiveControlMsg(msg *ControlMessage) {
	po.Control <- msg
}

// Fire accepts the value which is the sum of all inputs
// with the bias added; it sends the result of calling the sigmoid
// function on this value.
func (so *Output) Fire(val float64) {
	so.OutChan <- Sigmoid(val)
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
				// The sum of the (Xi * Wj)
				o.Fire(layerTotal + o.Bias)
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
