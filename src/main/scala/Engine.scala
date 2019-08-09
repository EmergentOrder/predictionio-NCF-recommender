package org.example.vanilla

import org.apache.predictionio.controller.EngineFactory
import org.apache.predictionio.controller.Engine
import java.util.concurrent.atomic.AtomicReference

case class Query(user: Option[Int], item: Option[Int], num: Int)

case class PredictedResult(targetScores: Seq[TargetScore])

case class TargetScore(target: Long,score: Double)

object VanillaEngine extends EngineFactory {

  val modelRef: AtomicReference[Array[Byte]] = new AtomicReference[Array[Byte]]
  val userdictRef: AtomicReference[java.util.HashMap[Int,Int]] = new AtomicReference[java.util.HashMap[Int,Int]]
  val itemdictRef: AtomicReference[java.util.HashMap[Int,Int]] = new AtomicReference[java.util.HashMap[Int,Int]]
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("algo" -> classOf[Algorithm]),
      classOf[Serving])
  }
}
