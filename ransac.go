package ransac

import (
	"math"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

type calcError func(*mat64.Vector, *mat64.Vector) float64
type calcModel func([]*mat64.Vector) *mat64.Vector

// Problem is the formulation of the ransac problem.
type Problem struct {
	data       []*mat64.Vector
	dataLength int
	fit        calcError
	model      calcModel
}

func (p *Problem) sample(sampleSize int) []*mat64.Vector {
	sample := make([]*mat64.Vector, sampleSize)
	currentSample := 0

	for currentSample < sampleSize {
		randomIndex := random(0, p.dataLength)
		exists := existsInData(sample, p.data[randomIndex])
		if !exists {
			sample[currentSample] = p.data[randomIndex]
			currentSample++
		}
	}
	return sample
}

func (p *Problem) calcModelError(model *mat64.Vector) float64 {
	var ssd float64
	for _, point := range p.data {
		error := p.fit(model, point)
		ssd += error * error
	}
	return ssd
}

// SetModel sets a function which describes the model.
func (p *Problem) SetModel(fn calcModel) {
	p.model = fn
}

// SetModelError sets the function, which calculates the error.
func (p *Problem) SetModelError(fn calcError) {
	p.fit = fn
}

// SetData sets the function, which sets the dataset.
func (p *Problem) SetData(data []*mat64.Vector) {
	p.data = data
	p.dataLength = len(data)
}

func (p *Problem) classifyInliers(model *mat64.Vector, sample []*mat64.Vector, maxError float64) ([]*mat64.Vector, []*mat64.Vector) {
	var inliers []*mat64.Vector
	var outliers []*mat64.Vector

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
func (p *Problem) Estimate(maxIterations, sampleSize int, inliersRatioLimit float64, maxError float64, improveWithConsensusSet bool) (*mat64.Vector, []*mat64.Vector, []*mat64.Vector, float64) {

	var bestInliers []*mat64.Vector
	var bestOutliers []*mat64.Vector
	var bestModel *mat64.Vector
	var bestError float64 = math.Inf(1)

	// Return Infinite Error, if datasize is smaller than sampleSize
	if p.dataLength < sampleSize {
		return bestModel, bestInliers, bestOutliers, bestError
	}

	dataLength := float64(p.dataLength)

	for iteration := 0; iteration <= maxIterations; iteration++ {
		sample := p.sample(sampleSize)
		model := p.model(sample)
		inliers, outliers := p.classifyInliers(model, sample, maxError)
		inliersRatio := float64(len(inliers)) / dataLength
		if inliersRatio >= inliersRatioLimit {
			candidateModel := model
			if improveWithConsensusSet {
				candidateModel = p.model(inliers)
			}
			candidateError := p.calcModelError(candidateModel)
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

func existsInData(data []*mat64.Vector, sample *mat64.Vector) bool {
	for _, point := range data {
		if point == sample {
			return true
		}
	}
	return false
}
