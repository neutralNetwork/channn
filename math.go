package channn

import (
	"math"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func Sigmoid(z float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -z))
}
