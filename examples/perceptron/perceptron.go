package main

import (
	"fmt"
	"github.com/neutralNetwork/channn"
)

// Here we implement a bitwise adder with NAND gates.
// This is an example from the book Neural Networks and Deep Learning,
// located here:
// http://neuralnetworksanddeeplearning.com/index.html
func main() {

	var inputOne = channn.MakePerceptronInput()
	var inputTwo = channn.MakePerceptronInput()

	var neuronOne = channn.MakePerceptronNeuron(3)
	var neuronTwo = channn.MakePerceptronNeuron(3)
	var neuronThree = channn.MakePerceptronNeuron(3)

	var outCarryBit = channn.MakePerceptronOutput(3)

	var outOne = channn.MakePerceptronOutput(3)

	// Connect input one to neuron one and two
	inputOne.ConnectNeurons(neuronOne)
	inputOne.ConnectNeurons(neuronTwo)
	inputOne.ResetAllWeights(-2)

	// Connect input two to neuron one and three
	inputTwo.ConnectNeurons(neuronOne)
	inputTwo.ConnectNeurons(neuronThree)
	inputTwo.ResetAllWeights(-2)

	// Connect neuron one to neuron two, three and the carry bit
	neuronOne.ConnectNeurons(neuronTwo)
	neuronOne.ConnectNeurons(neuronThree)
	neuronOne.ResetAllWeights(-2)

	neuronOne.ConnectOutput(outCarryBit)
	// Need to double weight since we can't connect
	// to the carry bit neuron twice.
	neuronOne.Control <- &channn.ControlMessage{
		Id: channn.SET_WEIGHT,
		Key: &outCarryBit.InChan,
		Value: -4.0,
	}

	// Connect neuron two to the output one
	neuronTwo.ConnectOutput(outOne)
	neuronTwo.ResetAllWeights(-2)

	// Connect neuron three to the output one
	neuronThree.ConnectOutput(outOne)
	neuronThree.ResetAllWeights(-2)

	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			fmt.Println("i", i)
			fmt.Println("j", j)
			inputOne.Fire(float64(i))
			inputTwo.Fire(float64(j))
			fmt.Println("outOne.GetResult()", outOne.GetResult())
			fmt.Println("outCarryBit.GetResult()", outCarryBit.GetResult())
		}
	}
}
