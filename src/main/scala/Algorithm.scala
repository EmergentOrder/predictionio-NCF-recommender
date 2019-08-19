package org.example.vanilla

import org.apache.predictionio.controller.P2LAlgorithm
import org.apache.predictionio.controller.Params

import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

import org.apache.predictionio.data.storage.Event
import org.apache.predictionio.data.store.LEventStore

import scala.util.Random
import scala.concurrent.duration.Duration
import collection.JavaConverters._

import scala.io.Source
import scala.reflect.io.Streamable

import zio.Task
import zio.DefaultRuntime
import org.emergentorder.onnxZIO.NCFZIO
import org.emergentorder.onnx.TensorFactory
import org.emergentorder.onnx.XInt

case class AlgorithmParams(appName: String, eventName: String) extends Params

class Algorithm(val ap: AlgorithmParams)
  // extends PAlgorithm if Model contains RDD[]
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  val runtime = new DefaultRuntime {}

  override
  def train(sc: SparkContext, data: PreparedData): Model = {

    val bytes = VanillaEngine.modelRef.get() //.map(x => Byte.unbox(x))
    val userIdsMap = VanillaEngine.userdictRef.get().asScala.toMap map {case (key, value) => (key.toLong, value.toLong)}
    val itemIdsMap = VanillaEngine.itemdictRef.get().asScala.toMap map {case (key, value) => (key.toLong, value.toLong)}
    new Model(bytes, userIdsMap, itemIdsMap)
  }

  override
  def predict(model: Model, query: Query): PredictedResult = {

    val eventName = ap.eventName
    val appName = ap.appName

    val ncf = new NCFZIO(model.model, model.userIdMap, model.itemIdMap)

    val timeout = 15000
    def recentEvents(queryEntityId: Option[String], queryTargetEntityId: Option[Option[String]]): Seq[Event] = try {
      LEventStore.find(
        appName = appName,
        entityType = Some("user"),
        entityId = queryEntityId,
        targetEntityType = Some(Some("item")),
        targetEntityId = queryTargetEntityId,
        eventNames = Some(Seq(eventName)),
        timeout = Duration(timeout, "millis")).toSeq
    } catch {
      case e: scala.concurrent.TimeoutException =>
        logger.error(s"Timeout on query recent events")
        Seq.empty[Event]
      case e: NoSuchElementException =>
        logger.info("User id not found")
        Seq.empty[Event]
      case e: Exception =>
        logger.error(s"Error reading recent events")
        Seq.empty[Event]
    }

    def inputsItemToUserQuery(itemId: Int) = {
      val numTargets = model.userIdMap.size
      //TODO: async
      val recentItemInteractedUsers = recentEvents(None, Some(Some("Event-" + itemId.toString))).map(x => x.entityId)
        .map(x => x.substring(5).toLong)

      val reverseUserIdMap = for ((k,v) <- model.userIdMap) yield (v, k)
      val rawCandidates = (0 until numTargets).toArray.map(x => reverseUserIdMap(x.toLong))
      //To subsample 
      //val rawCandidates = Seq.fill(numCandidates)(Random.nextInt(numTargets)).toArray.map(x => reverseUserIdMap(x.toLong))

      //TODO: move filtering to the end, to avoid varying shapes
      val filteredCandidates = (rawCandidates diff recentItemInteractedUsers).distinct

      val candidates = (filteredCandidates, Array(filteredCandidates.size)) 

      val itemInput = Task{
        (Array.fill[Long](filteredCandidates.size)(itemId), Array(filteredCandidates.size))
       }


      val userInput = Task{
        candidates
      }
      (userInput, itemInput, candidates)
    }

    def inputsUserToItemQuery(userId: Int) = {

      val numTargets = model.itemIdMap.size

      val recentUserInteractedItems = recentEvents(Some("User-" + userId.toString), None).map(x => x.targetEntityId)
        .map(x => x.map(y => y.substring(6).toLong)).flatten

      val reverseItemIdMap = for ((k,v) <- model.itemIdMap) yield (v, k)
      val rawCandidates = (0 until numTargets).toArray.map(x => reverseItemIdMap(x.toLong))
      //val rawCandidates = Seq.fill(numCandidates)(Random.nextInt(numTargets)).toArray.map(x => reverseUserIdMap(x.toLong))

      val filteredCandidates = (rawCandidates diff recentUserInteractedItems).distinct

      val inputSize = Array(filteredCandidates.size)
      val candidates = TensorFactory.getTensor(filteredCandidates, inputSize)

      val userInput = Task{
        TensorFactory.getTensor(Array.fill[Long](filteredCandidates.size)(userId), inputSize)
       }



      val itemInput = Task{
        candidates
      }
      (userInput, itemInput, candidates)
    }

    val (userInput, itemInput, candidates) = query.user match {
      case Some(x) => inputsUserToItemQuery(x)
      case None => query.item match {
        case Some(y) => inputsItemToUserQuery(y)
        case None => throw new RuntimeException("Must supply either user or item id")
      }
    } 

    def program = ncf.fullNCF(userInput, itemInput)

    val before = System.nanoTime
    val output = runtime.unsafeRun(program)
    val after = System.nanoTime

//    println("TIME " + (after-before))
    val targetOutputs = output._1.grouped(2).map(x => x(1)).toArray
      .zip(candidates._1)

    val sortedOutputs = targetOutputs.sortBy(_._1).reverse.take(query.num)
    val softmaxedSortedOutputs = sortedOutputs.map(x => TargetScore(target = x._2,
      score = scala.math.exp(x._1)/(scala.math.exp(x._1) + 1)) //Apply softmax to a single logit
    )

    PredictedResult(targetScores = softmaxedSortedOutputs)
  }
}

  case class Model(model: Array[Byte], userIdMap: Map[Long, Long], itemIdMap: Map[Long, Long])
