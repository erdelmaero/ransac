package ransac

import (
	"math"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func calcModelTest(sample []*mat64.Vector) *mat64.Vector {
	// sample is an array of points
	p1 := sample[0]
	p2 := sample[1]

	// x := p1.At(0, 0)
	m := (p2.At(1, 0) - p1.At(1, 0)) / (p2.At(0, 0) - p1.At(0, 0))
	b := p1.At(1, 0) - m*p1.At(0, 0)
	data := []float64{
		m,
		b,
	}

	model := mat64.NewVector(2, data)

	return model
}

func calcErrorTest(model, point *mat64.Vector) float64 {
	yEstimated := model.At(0, 0)*point.At(0, 0) + model.At(1, 0)
	ransacError := point.At(1, 0) - yEstimated
	return math.Abs(ransacError)
}

func TestRansac(t *testing.T) {

	p0Data := []float64{
		1,
		1,
	}
	p0 := mat64.NewVector(2, p0Data)

	p1Data := []float64{
		2,
		2,
	}
	p1 := mat64.NewVector(2, p1Data)

	p2Data := []float64{
		3,
		3,
	}
	p2 := mat64.NewVector(2, p2Data)

	// Outlier
	p3Data := []float64{
		4,
		10,
	}
	p3 := mat64.NewVector(2, p3Data)

	data := make([]*mat64.Vector, 4)

	data[0] = p0
	data[1] = p1
	data[2] = p2
	data[3] = p3

	prob := Problem{}
	prob.SetData(data)
	prob.SetModelError(calcErrorTest)
	prob.SetModel(calcModelTest)
	model, inliers, _, _ := prob.Estimate(1000, 2, 0.1, 0.1, false)

	if len(inliers) != 3 {
		t.Error("There should be 3 inliers there are", len(inliers))
	}

	if model == nil {
		t.Error("There should be a model found!")
	}

	if model.At(0, 0) != 1 || model.At(1, 0) != 0 {
		t.Error("Model should be y = 1*x + 0")
	}

}

func TestExistsInData(t *testing.T) {
	v := mat64.NewVector(3, []float64{1, 2, 3})

	v2 := mat64.NewVector(3, []float64{1, 5, 3})
	v3 := mat64.NewVector(3, []float64{1, 2, 4})
	v4 := mat64.NewVector(3, []float64{1, 2, 6})

	vecs := []*mat64.Vector{v2, v3, v4}
	if existsInData(vecs, v) {
		t.Error("v exists not in vecs but shows true")
	}

	vecs = append(vecs, v)
	if !existsInData(vecs, v) {
		t.Error("v exists in vecs but shows false")
	}
}
