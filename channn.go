package channn

type NeuronType int

const (
	_ = iota
	PERCEPTRON_TYPE NeuronType = 1 << iota
	SIGMOID_TYPE
)
