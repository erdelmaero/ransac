package ransac

import (
	"math/rand"
	"reflect"
)

type calcError func(map[string]float64, map[string]float64) float64
type calcModel func([]map[string]float64) map[string]float64

// Problem is the formulation of the ransac problem.
type Problem struct {
	data  []map[string]float64
	fit   calcError
	model calcModel
}

func (p Problem) sample(sampleSize int) []map[string]float64 {
	sample := make([]map[string]float64, sampleSize)
	currentSample := 0

	for currentSample < sampleSize {
		randomIndex := random(0, len(p.data))
		exists := existsInData(sample, p.data[randomIndex])
		if !exists {
			sample[currentSample] = p.data[randomIndex]
			currentSample++
		}
	}
	return sample
}

func (p Problem) calcModelError(model map[string]float64) float64 {
	var ssd float64
	for _, point := range p.data {
		error := p.fit(model, point)
		ssd += error * error
	}
	return ssd
}

// SetModel sets a function which describes the mode.
func (p *Problem) SetModel(fn calcModel) {
	p.model = fn
}

// SetModelError sets the function, which calculates the error.
func (p *Problem) SetModelError(fn calcError) {
	p.fit = fn
}

// SetData sets the function, which sets the dataset.
func (p *Problem) SetData(data []map[string]float64) {
	p.data = data
}

func (p Problem) classifyInliers(model map[string]float64, sample []map[string]float64, maxError float64) ([]map[string]float64, []map[string]float64) {
	var inliers []map[string]float64
	var outliers []map[string]float64

	for _, point := range p.data {
		if !existsInData(sample, point) {
			error := p.fit(model, point)
			if error <= maxError {
				inliers = append(inliers, point)
			} else {
				outliers = append(outliers, point)
			}
		}
	}
	return inliers, outliers
}

// Estimate does the actual work of fitting.
func (p Problem) Estimate(maxIterations, sampleSize int, inliersRatioLimit float64, maxError float64) (map[string]float64, []map[string]float64, []map[string]float64, float64) {
	var iteration int
	var bestInliers []map[string]float64
	var bestOutliers []map[string]float64
	var bestModel map[string]float64
	var bestError float64 = 9999999999

	for iteration <= maxIterations {
		sample := p.sample(sampleSize)
		model := p.model(sample)
		inliers, outliers := p.classifyInliers(model, sample, maxError)
		inliersRatio := float64(len(inliers)) / float64(len(p.data))
		if inliersRatio < inliersRatioLimit {
			candidateError := p.calcModelError(model)
			if candidateError < bestError {
				bestInliers = inliers
				bestOutliers = outliers
				bestModel = model
				bestError = candidateError
			}
		}
		iteration = iteration + 1
	}

	return bestModel, bestInliers, bestOutliers, bestError
}

func random(min, max int) int {
	return rand.Intn(max-min) + min
}

func existsInData(data []map[string]float64, sample map[string]float64) bool {
	for _, point := range data {
		if reflect.DeepEqual(point, sample) {
			return true
		}
	}
	return false
}
