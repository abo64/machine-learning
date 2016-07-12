package perceptron

import breeze.linalg._
import breeze.plot._
import PerceptronBreeze.DataSet
import PerceptronBreeze.Example
import PerceptronBreeze.Iterations
import PerceptronBreeze.Weight

class BreezeViz(ds: DataSet, w: Weight, iterations: Iterations) {
  private val f = Figure()
  private val p = f.subplot(0)
  private val x = linspace(0.0, 1.0)

  p.title = s"$iterations iterations"
  p.xlabel = "x axis"
  p.ylabel = "y axis"
  p.ylim(0, 1)

  def save = {
    plotDataSet
    plotWeight
    saveAs("learned.png")
  }

  private def plotWeight = {
    val y = x map { xi =>
      // w(0) + w(1)*x + w(2)*y = 0
      (-w(0) - w(1) * xi) / w(2)
    }
    p += plot(x, y)
  }

  private def plotDataSet = {
    def plotExample(e: Example) = {
      val (in, out) = e
      val (x, y) = (in(1), in(2))
      val d = 0.005d
      val color = if (out > 0) "green" else "red"
      p += plot(Seq(x - d, x + d), Seq(y - d, y + d), colorcode = color)
      p += plot(Seq(x - d, x + d), Seq(y + d, y - d), colorcode = color)
    }

    ds foreach plotExample
  }

  private def saveAs(fileName: String) = {
    f.saveas(fileName)
  }
//  f.drawPlots(g2d)
}