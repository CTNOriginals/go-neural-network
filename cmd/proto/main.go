package main

import (
	"fmt"
	"time"

	"github.com/CTNOriginals/go-neural-network/activators"
)

func main() {
	var startTime = time.Now()
	fmt.Printf("\n\n---- go-neural-network START %s ----\n", startTime.Format(time.TimeOnly))
	defer func() {
		fmt.Printf("\n---- go-neural-network END %s (%f) ----\n", startTime.Format(time.TimeOnly), time.Since(startTime).Seconds())
	}()

	var methods = []activators.Activator{
		activators.ReLU,
		activators.LeakyReLU,
		activators.Sigmoid,
	}
	var names = []string{
		"ReLU",
		"LeakyReLU",
		"Sigmoid",
	}

	var nums = []float64{0, 0.5, 1, -1, 2, 5, -5, 10, 100}

	for _, num := range nums {
		fmt.Printf("--- %.2f ----\n", num)

		for i, fn := range methods {
			fmt.Printf("%s:   \t %.2f\t%.2f\n", names[i], fn.Forward(num), fn.Backward(num))
		}
	}
}
