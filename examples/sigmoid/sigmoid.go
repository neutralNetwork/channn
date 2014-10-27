package main

import (
	"fmt"
	"github.com/neutralNetwork/channn"
)




func main() {

	var inputOne = channn.MakeInput()
	var inputTwo = channn.MakeInput()

	var neuronOne = channn.MakeNeuron(1)
	var neuronTwo = channn.MakeNeuron(1)
	var neuronThree = channn.MakeNeuron(1)

	var outCarryBit = channn.MakeSigmoidOutput(1)

	var outOne = channn.MakeSigmoidOutput(1)
