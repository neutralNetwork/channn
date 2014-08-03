package channn

// Control messages
const (
	CONNECT = iota
	DISCONNECT
	DESTROY
	RESET_WEIGHTS
	SET_WEIGHTS     // Set all weights
	SET_WEIGHT      // Set a single weight
	INCREMENT_INPUT // For signaling a new input
)

type ControlMessage struct {
	Id    int
	Key   interface{}
	Value interface{}
}
