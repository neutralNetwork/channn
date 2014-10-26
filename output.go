package channn

import (
	"fmt"
	"sync"
)

func MakePerceptronOutput(bias float64) *Output {
	z := int32(0)
	o := &Output{
		InChan:  make(chan float64),
		NumIn:   &z,
		Control: make(chan *ControlMessage),
		Bias:    bias,
		OutChan: make(chan float64),
		mutex:   &sync.Mutex{},
		nType:   PERCEPTRON_TYPE,
	}
	go o.ListenPerceptron()
	return o
}


// Output
type Output struct {
	InChan  chan float64
	NumIn   *int32
	Bias    float64
	Control chan *ControlMessage
	Result  float64
	OutChan chan float64
	mutex   *sync.Mutex
	nType    NeuronType
}
func (n *Output) String() string {
	return fmt.Sprintf("output %s", &n)
}

// FirePerceptron conditionally sends a value of 1.0 or 0.0
// to the output channel.
func (o *Output) FirePerceptron(val float64) {
	if val >= 1.0 {
		o.OutChan <- 1.0
	} else {
		o.OutChan <- 0.0
	}
}

// FireSigmoid accepts the value which is the sum of all inputs
// with the bias added; it sends the result of calling the sigmoid
// function on this value.
func (o *Output) FireSigmoid(val float64) {
	o.OutChan <- Sigmoid(val)
}


func (o *Output) GetResult() float64 {
	return <-o.OutChan
}

func (o *Output) ListenPerceptron() {
	var counter = *o.NumIn
	var layerTotal float64
	for {
		select {
		case inVal := <-o.InChan:
			layerTotal += inVal
			counter--

			if counter == 0 {
				// The sum of the SUM(XiWj) + Bias
				o.FirePerceptron(layerTotal + o.Bias)
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
