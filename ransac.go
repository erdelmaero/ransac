package ransac

import (
	"math"
	"math/rand"
)

type calcError func(map[string]float64, [2]float64) float64
type calcModel func([][2]float64) map[string]float64

// Problem is the formulation of the ransac problem.
type Problem struct {
	data  [][2]float64
	fit   calcError
	model calcModel
}

func (p Problem) sample(sampleSize int) [][2]float64 {
	sample := make([][2]float64, sampleSize)
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
func (p *Problem) SetData(data [][2]float64) {
	p.data = data
}

func (p Problem) classifyInliers(model map[string]float64, sample [][2]float64, maxError float64) ([][2]float64, [][2]float64) {
	var inliers [][2]float64
	var outliers [][2]float64

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
func (p Problem) Estimate(maxIterations, sampleSize int, inliersRatioLimit float64, maxError float64, improveWithConsensusSet bool) (map[string]float64, [][2]float64, [][2]float64, float64) {
	var bestInliers [][2]float64
	var bestOutliers [][2]float64
	var bestModel map[string]float64
	var bestError float64 = math.Inf(1)

	for iteration := 0; iteration <= maxIterations; iteration++ {
		sample := p.sample(sampleSize)
		model := p.model(sample)
		inliers, outliers := p.classifyInliers(model, sample, maxError)
		inliersRatio := float64(len(inliers)) / float64(len(p.data))
		if inliersRatio >= inliersRatioLimit {
			candidateModel := model
			if improveWithConsensusSet {
				candidateModel = p.model(inliers)
			}
			candidateError := p.calcModelError(model)
			if candidateError < bestError {
				bestInliers = inliers
				bestOutliers = outliers
				bestModel = candidateModel
				bestError = candidateError
			}
		}
	}

	return bestModel, bestInliers, bestOutliers, bestError
}

func random(min, max int) int {
	return rand.Intn(max-min) + min
}

func existsInData(data [][2]float64, sample [2]float64) bool {
	for _, point := range data {
		if point[0] == sample[0] && point[1] == sample[1] {
			return true
		}
	}
	return false
}
