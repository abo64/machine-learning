package perceptron

import PerceptronPlainScala._
import scala.annotation.tailrec
import scala.util.Random

class PerceptronPlainScala(d: Dimension, f: TargetFunction, dataSetSize: DataSetSize,
    random: Random = Random.self)
{
  def genDataSet: DataSet = {
    def genInput: Input = (Vector.fill(d)(random.nextDouble)) .+: (1d)
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

  val StartWeight: Weight = Vector.fill(d + 1)(0d)

  // PLA
  def learn(dataSet: DataSet): (Weight, Iterations) = {
    val h: Hypothesis = (w, x) =>
      math.signum((w zip x) map { case (wi, xi) => wi * xi } sum) toInt

    def update(w: Weight, example: Example): Weight = {
      val (x, y) = example
      (w zip x) map { case (wi, xi) => wi + y * xi }
    }

    def pickMisclassified(w: Weight): Option[Example] =
      dataSet find { case (x, y) => h(w, x) != y }

    @tailrec
    def loop(w: Weight, misclassified: Option[Example], iterations: Iterations): (Weight, Iterations) =
      misclassified match {
        case None => (w, iterations)
        case Some(example) =>
          val newW = update(w, example)
          loop(newW, pickMisclassified(newW), iterations + 1)
    }

    loop(StartWeight, pickMisclassified(StartWeight), 0)
  }
}

object PerceptronPlainScala {
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
    val f: TargetFunction = {
      case Vector(1d, x1, x2) => if (x1 <= x2) 1 else -1
    }

    val perceptron = new PerceptronPlainScala(2, f, 2000)
    val dataSet = perceptron.genDataSet
    println(dataSet count {case (_,b) => b > 0})
    val (w,i) = perceptron.learn(dataSet)
    println(w)
    println(s"$i iterations")
  }
}