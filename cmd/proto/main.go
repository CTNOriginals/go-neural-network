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

	var nums = []float64{0, 0.5, 1, -1, 2, 5, -5, 10, 100}

	for _, num := range nums {
		activators.PrintResults(num)
	}
}
