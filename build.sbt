name := "template-scala-NCF-recommender"

resolvers += Resolver.mavenLocal
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
scalaVersion := "2.11.12"
libraryDependencies ++= Seq(
  "com.github.EmergentOrder" %% "onnx-scala-zio" % "0.2.0-SNAPSHOT",
  "org.apache.predictionio" %% "apache-predictionio-core" % "0.14.0" % "provided",
  "org.apache.spark"        %% "spark-mllib"              % "2.4.3" % "provided",
  "org.scalatest"           %% "scalatest"                % "3.0.5" % "test")

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", "MANIFEST.MF") => MergeStrategy.discard
  case _                                => MergeStrategy.last
}

// SparkContext is shared between all tests via SharedSingletonContext
parallelExecution in Test := false
