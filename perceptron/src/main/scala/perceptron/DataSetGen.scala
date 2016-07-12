package perceptron

import scala.annotation.tailrec

class DataSetGen[Input, Output](genInput: => Input, classify: Input => Output) {
  type Example = (Input, Output)
  type DataSet = Set[Example]

  def genDataSet(howMany: Int): DataSet = {
    def genExample: Example = {
      val input = genInput
      (input, classify(input))
    }

    @tailrec
    def loop(result: DataSet): DataSet =
      if (result.size == howMany) result
      else loop(result + genExample)

    loop(Set())
  }
}
