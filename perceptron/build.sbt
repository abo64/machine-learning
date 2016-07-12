name := """perceptron"""

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies  ++= Seq(
  // other dependencies here
  "org.scalanlp" %% "breeze" % "0.12",
  // native libraries are not included by default. add this if you want them (as of 0.7)
  // native libraries greatly improve performance, but increase jar sizes. 
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.12",
  // the visualization library is distributed separately as well. 
  // It depends on LGPL code.
  "org.scalanlp" %% "breeze-viz" % "0.12")

  // To depend on snapshot versions, use:
  //    "org.scalanlp" %% "breeze" % "latest.integration",
//)



//resolvers ++= Seq(
  // other resolvers here
  // if you want to use snapshot builds (currently 0.13-0598e003cfa7f00f76919aa556009ad6d4fc1332-SNAPSHOT), use this.
//  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
//  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
//)

EclipseKeys.withSource := true
