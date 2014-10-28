package channn

type NeuronType int

const (
	PERCEPTRON_TYPE NeuronType = iota
	SIGMOID_TYPE
	INPUT_TYPE
	OUTPUT_TYPE
)


type ChanNeuron interface {
	GetInChanPtr() *chan float64
	ReceiveControlMsg(*ControlMessage)
}
