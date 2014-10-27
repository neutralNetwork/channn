package channn

type NeuronType int

const (
	PERCEPTRON_TYPE NeuronType = iota
	SIGMOID_TYPE
	INPUT_TYPE
	OUTPUT_TYPE
)


type ChanNeuron interface {
	Fire(float64)
	GetType() NeuronType
}
