package main

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

func main() {
	start := time.Now()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║       GO-NEURAL-NETWORK SANDBOX DEMONSTRATIONS             ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	demos := []func(){
		RunXorGateDemo,
		RunLogicGatesDemo,
		RunPatternDemo,
		RunFunctionApproxDemo,
		RunSimpleClassifierDemo,
	}

	var args = os.Args[1:]
	fmt.Printf("args: %v\n", args)
	if len(args) > 0 {
		var target, _ = strconv.Atoi(args[0])
		demos[target]()
		return
	}

	for i, demo := range demos {
		fmt.Printf("\n┌──────────────────────────────────────────────────────────────┐\n")
		fmt.Printf("│ Demo %d/%d: %-45s│\n", i+1, len(demos), getDemoName(i))
		fmt.Printf("└──────────────────────────────────────────────────────────────┘\n")
		demo()
	}

	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Printf("║ All demos completed in %.2fs                                 ║\n", time.Since(start).Seconds())
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
}

func getDemoName(i int) string {
	names := []string{
		"XOR Gate",
		"Logic Gates (AND, OR, NAND)",
		"Pattern Learning",
		"Function Approximation (Sine)",
		"Simple Classifier",
	}
	if i < len(names) {
		return names[i]
	}
	return "Unknown"
}

