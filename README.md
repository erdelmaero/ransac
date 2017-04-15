# ransac

RANSAC implementation written in Go


### Getting Started

This example shows how to fit a line into 2D points.
```
// Import package
import (
	"fmt"
	"math"

	"github.com/erdelmaero/ransac"
)

// Define the model using maps
func calcModel(sample []map[string]float64) map[string]float64 {
  // sample is an array of points
	p1 := sample[0]
	p2 := sample[1]

  // m and b define the model of a line.
	m := (p2["y"] - p1["y"]) / (p2["x"] - p1["x"])
	b := p1["y"] - m*p1["x"]

  // return the calculated model
	model := make(map[string]float64, 2)
	model["m"] = m
	model["b"] = b
	return model
}

// Define function, which calculates the error.
func calcError(model, point map[string]float64) float64 {
	yEstimated := model["m"]*point["x"] + model["b"]
	ransacError := point["y"] - yEstimated
	return math.Abs(ransacError)
}

// In Main define some test points

func main() {
  // Slice where we put the points.
	data := make([]map[string]float64, 3)

  // Points as maps with x, y. Can have n dimensions.
	p0 := make(map[string]float64)
	p0["x"] = 0
	p0["y"] = 0.2

	p1 := make(map[string]float64)
	p1["x"] = 2
	p1["y"] = 1.9

	p2 := make(map[string]float64)
	p2["x"] = 2.1
	p2["y"] = 1.85

  // Append the points to slice:
	data[0] = p0
	data[1] = p1
	data[2] = p2

  // Define ransac problem:
	prob := ransac.Problem{}
	prob.SetData(data)
	prob.SetModelError(calcError)
	prob.SetModel(calcModel)

  // Estimate model. Estimate takes maxIterations, sampleSize (two to define a 2D line) and inliersRatioLimit.
  // The Last Parameter is the maxError.
	model, _, _, err := prob.Estimate(1000, 2, 0.5, 0.1)
	fmt.Println(model, err)
}

```
