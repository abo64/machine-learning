package perceptron

import scala.annotation.tailrec
import breeze.linalg._
import breeze.plot._
import PerceptronBreeze._
import scala.util.Random

class PerceptronBreeze(d: Dimension, f: TargetFunction, dataSetSize: DataSetSize)
{
  def genDataSet: DataSet = {
    def genInput: Input = {
      val v = DenseVector.rand[Double](d + 1)
      v(0) = 1d
      v
    }

    def genExample: Example = {
      val input = genInput
      (input, f(input))
    }

    @tailrec
    def loop(result: DataSet): DataSet =
      if (result.size == dataSetSize) result
      else loop(result + genExample)

    loop(Set())
  }

  val StartWeight: Weight =
//    Vector.rand(d + 1)
//    Vector.fill(d + 1)(1d)
    Vector.zeros(d + 1)

  // PLA
  def learn(dataSet: DataSet): (Weight, Iterations) = {
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
    def loop(w: Weight, misclassified: Option[Example], iterations: Iterations): (Weight, Iterations) =
      misclassified match {
        case None =>
          showStatus(w, iterations, min(dataSet.size, 100))
          (w, iterations)
        case Some(example) =>
          val newW = update(w, example)
          if (iterations % 10 == 0 && iterations <= 100) showStatus(w, iterations)
          loop(newW, pickMisclassified(newW), iterations + 1)
    }

    loop(StartWeight, pickMisclassified(StartWeight), 0)
  }
}

object PerceptronBreeze {
  type Dimension = Int
  type Input = Vector[Double]
  type Output = Int // +1 or -1
  type TargetFunction = Input => Output
  type Example = (Input, Output)
  type DataSet = Set[Example]
  type DataSetSize = Int
  type Weight = Vector[Double]
  type Hypothesis = (Weight, Input) => Output
  type Iterations = Int

  def main(args: Array[String]): Unit = {
    val f: TargetFunction = { v =>
      if (v(1) <= v(2)) 1 else -1
    }
    val g: TargetFunction = { v =>
      if (v(1) <= v(2)/2) 1 else -1
    }

    val perceptron = new PerceptronBreeze(2, f, 20)
    val dataSet = perceptron.genDataSet
    println(dataSet count {case (_,b) => b > 0})
    val (w,i) = perceptron.learn(dataSet)
    println(w)
    println(s"$i iterations")
  }
}