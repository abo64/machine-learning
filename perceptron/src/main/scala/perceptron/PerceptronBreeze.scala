package perceptron

import scala.annotation.tailrec
import breeze.linalg._
import breeze.plot._
import PerceptronBreeze._
import scala.util.Random

object PerceptronBreeze {

  type Dimension = Int
  type Input = Vector[Double]
  type Output = Int // +1 or -1
  type Classifier = Input => Output
  type TargetFunction = Classifier
  type Example = (Input, Output)
  type DataSet = Set[Example]
  type DataSetSize = Int
  type Weight = Vector[Double]
  type Hypothesis = (Weight, Input) => Output
  type Iterations = Int

  // PLA
  def learn(dataSet: DataSet, startWeight: Weight): (Weight, Iterations, Classifier) = {
    val h: Hypothesis = (w, x) =>
      math.signum(w.t * x) toInt

    def update(w: Weight, example: Example): Weight = {
      val (x, y) = example
      val yx = x map (_ * y)
      w :+ yx
    }

    def pickMisclassified(w: Weight): Option[Example] =
      dataSet find { case (x, y) => h(w, x) != y }

    def showStatus(w: Weight, iterations: Iterations, howManyData: Int = 20) = {
      println(s"$iterations iterations: $w")
      val dataToShow = Random.shuffle(dataSet) take howManyData
      new BreezeViz(dataToShow, w, iterations).save
    }

    @tailrec
    def loop(w: Weight, misclassified: Option[Example],
        iterations: Iterations): (Weight, Iterations, Classifier) =
      misclassified match {
        case None =>
          showStatus(w, iterations, min(dataSet.size, 100))
          (w, iterations, h(w, _))
        case Some(example) =>
          val newW = update(w, example)
          if (iterations % 10 == 0 && iterations <= 100) showStatus(w, iterations)
          loop(newW, pickMisclassified(newW), iterations + 1)
    }

    loop(startWeight, pickMisclassified(startWeight), 0)
  }

  def predict(dataSet: DataSet, classifier: Classifier): (Int, Int) = {
    val misclassified = dataSet count { case (in, expectedOut) =>
      classifier(in) != expectedOut
    }
    (misclassified, dataSet.size)
  }

  def main(args: Array[String]): Unit = {
    val f: TargetFunction = { v =>
      if (v(1) <= v(2)) 1 else -1
    }
    val g: TargetFunction = { v =>
      if (v(1) <= v(2)/2) 1 else -1
    }

    val d: Dimension = 2

    val startWeight: Weight =
//      Vector.rand(d + 1)
//      Vector.fill(d + 1)(1d)
      Vector.zeros(d + 1)

    def genInput: Input = {
      val v = DenseVector.rand[Double](d + 1)
      v(0) = 1d
      v
    }

    val dataSetGen = new DataSetGen(genInput, f)
    val perceptron = PerceptronBreeze
    val trainingSet = dataSetGen.genDataSet(20)
    val ones = trainingSet count {case (_,b) => b > 0}
    println(s"$ones/${trainingSet.size}")
    val (w,i, classifier) = perceptron.learn(trainingSet, startWeight)
    println(w)
    println(s"$i iterations")

    val testSet = dataSetGen.genDataSet(100)
    val (misclassified, total) = perceptron.predict(testSet, classifier)
    val hits = total - misclassified
    val successRate = math.round(hits * 10000d / total) / 100d
    println(s"prediction result: $successRate% ($hits/$total)")
  }
}